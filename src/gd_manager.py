import os
import time
import random
import queue
import threading
import numpy
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# .envファイルから環境変数を読み込むため (必ずファイルの冒頭に置く)
from dotenv import load_dotenv 
load_dotenv() 

# Google Cloud Speech-to-Text (ASR) APIクライアント
from google.cloud import speech_v1p1beta1 as speech 
# Google Cloud Text-to-Speech (TTS) APIクライアント
from google.cloud import texttospeech_v1beta1 as texttospeech 

# Google Gemini APIクライアント (追加)
import google.generativeai as genai 

import pyaudio # 音声入力（マイク）と音声出力（スピーカー）を制御
import wave # WAVファイルの読み書き（一時ファイルの保存用）
import numpy as np # 音声データ処理（無音生成、形式変換など）

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, 
    QMainWindow, QPushButton, QStackedWidget, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer
from PySide6.QtGui import QTextCursor  # 追加: 自動スクロール用

# 新しいUIコンポーネントをインポート
from gd_ui_components import (
    UserInputScreen, GDScreen, FeedbackScreen, 
    GroupSelectionScreen, GDStartConfirmScreen, ControlGroupAfterFirstScreen,
    SpeakerCheckScreen
)

# --- システム共通設定 ---
# gd_manager.py は src/ ディレクトリにあるため、プロジェクトルートパスを正確に取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 開発モード設定 ===
# 開発中は True、本番は False に設定
DEV_MODE = False  # 実験用: False（開発モード: 時間を10秒に短縮、自動開始は無効）
DEV_DEFAULT_USERNAME = "テスト"  # 開発モード時のデフォルト名字
DEV_THINKING_SECONDS = 0  # 開発モード時の思考時間（0=スキップ、現在は使用されていません）
SKIP_INTRO = False  # 実験用: False（開発モードでも最初から最後までストーリーを確認するためFalseのまま）

# === テストモード設定 ===
# GD時間を2分に設定し、その他（読書時間・思考時間）をスキップするテスト用設定
TEST_MODE = False  # テスト用: TrueにするとGD時間2分、その他スキップ
# ==================

# 音声I/Oに関する共通設定
RATE = 16000    # サンプリングレート (Hz) - Google ASR/TTSの推奨値
# CHUNK = 1024     # 音声データを処理する際のバッファサイズ（フレーム数）。PyAudioで一度に読み書きする単位。
CHUNK = int(RATE / 10)  # 100ms ごとのデータ塊
FORMAT = pyaudio.paInt16    # 音声データのフォーマット。16ビット整数。
CHANNELS = 1    # 音声チャンネル数。1はモノラル。

# 一時的な音声ファイルを保存するパス (audio_dataフォルダはプロジェクトルート直下)
USER_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "user_input.wav")
AI_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "ai_output.wav")

# LLMとTTSの言語・モデル設定
LANGUAGE_CODE_ASR = "ja-JP" # ASR（音声認識）で認識する言語
LANGUAGE_CODE_TTS = "ja-JP" # TTS（音声合成）で生成する言語
DEFAULT_AI_VOICE_NAME = "ja-JP-Wavenet-C" # AIの声のデフォルト設定（Google TTSのボイス名）
# LLM_MODEL を Gemini のモデル名に変更
# GD中の対話・採点用（高速・低コスト）
GEMINI_MODEL = "gemini-2.5-flash"
# フィードバック生成専用（高精度・事後処理）
# 実験ではGD中より多少遅くても良い前提で、より高性能な 2.5 Pro を使用
GEMINI_FEEDBACK_MODEL = "gemini-2.5-pro"

# 使用済みテーマを保存するファイルパス
USED_THEMES_FILE = "used_themes.txt"

# --- アナウンス専用ヘルパー関数 ---
def _announce_system_message(text):
    """システムアナウンスを再生する（manager未初期化時でも使用可能）"""
    if not text:
        return
    
    try:
        from google.cloud import texttospeech_v1beta1 as texttospeech
        import pyaudio
        import numpy as np
        
        # マークダウン記号を除去
        import re
        cleaned_text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        cleaned_text = re.sub(r'\*([^\*]+)\*', r'\1', cleaned_text)
        cleaned_text = re.sub(r'__([^_]+)__', r'\1', cleaned_text)
        cleaned_text = re.sub(r'_([^_]+)_', r'\1', cleaned_text)
        cleaned_text = re.sub(r'#+\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned_text)
        cleaned_text = re.sub(r'`([^`]+)`', r'\1', cleaned_text)
        
        tts_client = texttospeech.TextToSpeechClient()
        p_audio = pyaudio.PyAudio()
        
        voice_name = "ja-JP-Neural2-D"
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_TTS,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            speaking_rate=1.2,
            pitch=0.0
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_content = response.audio_content
        
        stream = p_audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        stream.start_stream()
        
        # 無音データを先頭に追加
        silence_chunks = 3
        silence_data = np.zeros(CHUNK * silence_chunks, dtype=np.int16)
        audio_data = np.frombuffer(audio_content, dtype=np.int16)
        combined_audio = np.concatenate([silence_data, audio_data])
        total_frames = len(combined_audio)
        
        for i in range(0, total_frames, CHUNK):
            chunk_data = combined_audio[i:i+CHUNK]
            if len(chunk_data) < CHUNK:
                chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), mode='constant')
            stream.write(chunk_data.tobytes())
        
        stream.stop_stream()
        stream.close()
        p_audio.terminate()
        
        print(f"[システムアナウンス]: {text}")
    except Exception as e:
        print(f"[警告]: システムアナウンスの再生に失敗しました: {e}")

# 実験用固定テーマ
THEME_ROUND_1 = """学園祭模擬店の売上向上施策

あなたはゼミの模擬店リーダーです。昨年の「焼きそば」は売上が伸び悩み、利益がほとんど出ませんでした。今年の学園祭では「利益を昨年の1.5倍にする」ことが目標です。これから10分間で、メンバー（AI）と議論し、目標達成のための「具体的な施策を1つ」決定してください。※予算には限りがあります。"""

THEME_ROUND_2 = """オープンキャンパスの来場者数増加施策

あなたはオープンキャンパスの学生リーダーです。近年、高校生の来場者数が減少傾向にあり、大学側から対策を求められています。今年の開催に向けて、「来場者を確実に増やすための目玉企画」を1つ決定してください。これから10分間で、メンバー（AI）と議論し、結論を出してください。※学生スタッフだけで実施できる内容に限ります。"""

def get_fixed_gd_theme(round_number):
    """
    実験用の固定テーマを返す。
    
    Args:
        round_number: 1 または 2
    
    Returns:
        str: 固定テーマのテキスト
    """
    if round_number == 1:
        return THEME_ROUND_1
    elif round_number == 2:
        return THEME_ROUND_2
    else:
        raise ValueError(f"無効なラウンド番号: {round_number} (1または2を指定してください)")

def get_all_gd_themes():
    """
    すべてのGDテーマリストを返す。
    """
    return [
        # IT・テック系
        "都心の駅の混雑を緩和するためのITソリューションを提案してください。",
        "オンライン会議のコミュニケーション不足を解消する新しいアプリを企画してください。",
        "AI技術を活用して、高齢者の買い物の課題を解決するサービスを考案してください。",
        "AIを活用して、地方の観光地が抱える人手不足問題を解決するためのアイデアを提案してください。",

        # マーケティング・ビジネス戦略系
        "少子化が進む中で、遊園地の入場者数を増やすためのマーケティング戦略を立案してください。",
        "若者のテレビ離れが進む中、若者をターゲットにした新しいテレビ番組の企画を提案してください。",
        "地方の特産品を使った商品を、首都圏の消費者に届けるための販売戦略を考えてください。",

        # 社会問題・環境系
        "フードロスを削減するための新しいビジネスモデルを提案してください。",
        "都心のオフィス街におけるヒートアイランド現象を緩和するための、新しい都市計画を提案してください。",
        "地域のコミュニティバスの利用者を増やすための施策を提案してください。",
        "プラスチックごみの削減を促す、自治体と住民が協力する仕組みを提案してください。",

        # 教育・キャリア系
        "新卒者の離職率を低下させるための、企業が取り組むべき新しい研修プログラムを考案してください。",

        # ヘルスケア・福祉系
        "高齢者が安心して一人暮らしを続けられるような、地域社会と連携した新しい見守りサービスを企画してください。",

        # エンターテイメント・観光系
        "SNSのフォロワーを増やすための、美術館や博物館が取り組むべき新しいプロモーション戦略を提案してください。",

        # 地域社会・行政系
        "地域の伝統的な商店街を活性化させるための、新しいビジネスモデルを提案してください。",

        # ライフスタイル・小売系
        "実店舗の書店が、Amazonなどのオンライン書店に負けないための新しい集客戦略を提案してください。",
        "一人暮らしの若者が、健康的な食生活を維持するための新しい食品サービスを企画してください。",
    ]

def load_used_themes():
    """
    使用済みテーマをファイルから読み込む。
    """
    if not os.path.exists(USED_THEMES_FILE):
        return set()
    
    try:
        with open(USED_THEMES_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"使用済みテーマの読み込みエラー: {e}")
        return set()

def save_used_theme(theme):
    """
    使用したテーマをファイルに保存する。
    """
    try:
        with open(USED_THEMES_FILE, "a", encoding="utf-8") as f:
            f.write(theme + "\n")
    except Exception as e:
        print(f"使用済みテーマの保存エラー: {e}")

def reset_used_themes():
    """
    使用済みテーマの履歴をリセットする。
    """
    try:
        if os.path.exists(USED_THEMES_FILE):
            os.remove(USED_THEMES_FILE)
            print("[システム]: 使用履歴をリセットしました。")
    except Exception as e:
        print(f"使用履歴のリセットエラー: {e}")

def get_random_gd_theme():
    """
    未使用のGDテーマからランダムに選択して返す。
    すべてのテーマが使用済みの場合は、使用履歴をリセットして最初から再利用する。
    """
    all_themes = get_all_gd_themes()
    used_themes = load_used_themes()
    
    # 未使用のテーマを取得
    available_themes = [theme for theme in all_themes if theme not in used_themes]
    
    # すべて使用済みの場合はリセット
    if not available_themes:
        print("\n[システム]: すべてのテーマを使用しました。使用履歴をリセットします。")
        reset_used_themes()
        used_themes = set()
        available_themes = all_themes
    
    # ランダムに選択
    selected_theme = random.choice(available_themes)
    
    # 使用済みとして保存
    save_used_theme(selected_theme)
    
    print(f"[システム]: 残り未使用テーマ数: {len(available_themes) - 1}/{len(all_themes)}")
    
    return selected_theme

class MicrophoneStream(object):
    """マイクからの録音ストリームをジェネレーターとして開くクラス。"""

    def __init__(self, rate, chunk, timeout=8, p_audio=None, speaking_callback=None):
        self._rate = rate
        self._chunk = chunk
        self._timeout = timeout # タイムアウト設定
        self._buff = queue.Queue()
        self.closed = True
        self.last_audio_time = time.time() 
        self.last_speech_time = time.time()
        self.speaking = False
        self._audio_interface = p_audio # 渡されたp_audioインスタンスを使用
        self._speaking_callback = speaking_callback  # 音声入力開始時のコールバック
        self._callback_fired = False  # コールバック発火フラグ

    def __enter__(self):
        # self._audio_interface = pyaudio.PyAudio() # 削除
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # APIはモノラル（1チャンネル）のみをサポート
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # バッファにデータを非同期的に入力するためのコールバック関数
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # ジェネレーターに終了を通知
        self._buff.put(None)
        # self._audio_interface.terminate() # 削除

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """オーディオストリームからバッファにデータを継続的に収集する。"""

        # データをnumpy配列に変換して音量をチェック
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        volume_threshold = 500  # 音量閾値 (この値は環境によって調整可能)

        # まずは無条件にデータをキューにプットする (APIがリアルタイムを期待するため)
        self._buff.put(in_data)

        # 音量が閾値を超えているかチェック
        if np.max(np.abs(audio_data)) > volume_threshold:
            self.speaking = True
            self.last_speech_time = time.time()
            # 音声入力開始時にコールバックを呼び出す（1回のみ）
            if not self._callback_fired and self._speaking_callback:
                self._speaking_callback()
                self._callback_fired = True
        else:
            # 音量が閾値以下の場合、発話中フラグをリセット
            self.speaking = False

            # タイムアウトが設定されている場合のみ、無音検知を行う
            if self._timeout is not None:
                if not self.speaking and time.time() - self.last_speech_time > self._timeout:
                    self._buff.put(None)
                    self.closed = True

        # 重要: PyAudio のコールバックは必ずタプル (out_data, flag) を返す必要があります
        return (None, pyaudio.paContinue)
            
    def generator(self):
        while not self.closed:
            # ★タイムアウトを削除
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            
            # バッファに残っている他のデータをすべて消費
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
                
            yield b"".join(data)

# --- Zoom風GD UIメインウィンドウクラス ---
class GDReportWindow(QMainWindow):
    """実験用GDシステム - 画面遷移管理"""
    start_gd_requested = pyqtSignal(str, int)  # ユーザー名とラウンド番号を送信
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDシミュレーター - 実験用")
        self.setGeometry(100, 100, 1200, 800)
        
        # QStackedWidgetで画面切り替え
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 画面を作成
        self.group_selection_screen = GroupSelectionScreen()
        self.speaker_check_screen = SpeakerCheckScreen()
        self.user_input_screen = UserInputScreen()
        self.gd_start_confirm_screen = GDStartConfirmScreen()
        self.gd_screen = GDScreen()
        self.feedback_screen = FeedbackScreen()
        self.control_after_first_screen = ControlGroupAfterFirstScreen()
        
        # 画面を追加（インデックスを定義）
        self.stacked_widget.addWidget(self.group_selection_screen)      # index 0
        self.stacked_widget.addWidget(self.speaker_check_screen)        # index 1: スピーカーチェック
        self.stacked_widget.addWidget(self.user_input_screen)           # index 2
        self.stacked_widget.addWidget(self.gd_start_confirm_screen)     # index 3
        self.stacked_widget.addWidget(self.gd_screen)                   # index 4
        self.stacked_widget.addWidget(self.feedback_screen)             # index 5
        self.stacked_widget.addWidget(self.control_after_first_screen)  # index 6
        
        # シグナル接続
        self.group_selection_screen.group_selected.connect(self._on_group_selected)
        self.speaker_check_screen.speaker_check_completed.connect(self._on_speaker_check_completed)
        self.user_input_screen.system_start_requested.connect(self._on_username_entered)
        self.gd_start_confirm_screen.confirmed.connect(self._on_gd_start_confirmed)
        self.gd_start_confirm_screen.cancelled.connect(self._on_gd_start_cancelled)
        self.gd_start_confirm_screen.thinking_timeout.connect(self._on_thinking_timeout)
        self.feedback_screen.next_gd_requested.connect(self._on_next_gd_requested)
        self.feedback_screen.reading_timeout.connect(self._on_reading_timeout_feedback)
        self.control_after_first_screen.reading_timeout.connect(self._on_reading_timeout_control)
        
        # 状態管理
        self.current_username = ""
        self._announcement_in_progress = False  # アナウンス重複防止フラグ  # 名字のみ（GD内で使用）
        self.current_fullname = ""  # フルネーム（保存時に使用）
        self.experiment_group = None  # "experimental" または "control"
        self.current_gd_round = 1  # 1 または 2
        self.first_gd_feedback = {}  # 1回目のフィードバック（実験群のみ）
        
        # 初期画面を表示
        self.stacked_widget.setCurrentIndex(0)  # 実験群/統制群選択画面
    
    def _on_group_selected(self, group):
        """実験群/統制群選択 → スピーカーチェック画面へ"""
        self.experiment_group = group
        self.stacked_widget.setCurrentIndex(1)  # スピーカーチェック画面へ
        QApplication.processEvents()
    
    def _on_speaker_check_completed(self):
        """スピーカーチェック完了 → ユーザー名入力画面へ"""
        self.stacked_widget.setCurrentIndex(2)  # ユーザー名入力画面へ
        QApplication.processEvents()
        # システムアナウンス: 名前を入力してください
        if not self._announcement_in_progress:
            self._announcement_in_progress = True
            announcement = "名字と名前を入力してください。"
            # アナウンスは非同期で実行（UIをブロックしない）
            from threading import Thread
            def announce_and_reset():
                _announce_system_message(announcement)
                self._announcement_in_progress = False
            thread = Thread(target=announce_and_reset, daemon=True)
            thread.start()
    
    def _on_username_entered(self, lastname, firstname):
        """ユーザー名入力 → 1回目GD開始確認画面へ"""
        self.current_username = lastname  # 名字のみ（GD内で使用）
        self.current_fullname = lastname + firstname  # フルネーム（保存時に使用）
        self.current_gd_round = 1
        # 1回目GD開始確認画面を表示
        # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 2分
        if TEST_MODE:
            thinking_time_text = "1秒間"
            thinking_seconds = 1
        elif DEV_MODE:
            thinking_time_text = "10秒間"
            thinking_seconds = 10
        else:
            thinking_time_text = "2分間"
            thinking_seconds = 120
        self.gd_start_confirm_screen.set_message(
            f"これから1回目のグループディスカッションを開始します。\n手元の「①」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        )
        self.stacked_widget.setCurrentIndex(3)  # GD開始確認画面へ
        # システムアナウンス: 画面表示時にアナウンス
        announcement = f"これから1回目のグループディスカッションを開始します。手元の「①」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        # アナウンスは非同期で実行（UIをブロックしない）
        from threading import Thread
        from PySide6.QtCore import QTimer
        thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
        thread.start()
        # アナウンス再生後に思考時間タイマーを開始（テキストの長さから再生時間を推定）
        # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
        estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
        QTimer.singleShot(int(estimated_duration * 1000), lambda: self.gd_start_confirm_screen.start_thinking_time(thinking_seconds))
    
    def _on_gd_start_confirmed(self):
        """GD開始確認 → GD進行画面へ"""
        # 画面遷移を確実にする
        self.stacked_widget.setCurrentIndex(4)  # GD画面へ
        QApplication.processEvents()  # 画面更新を確実に実行
        
        # ローディング表示を開始
        self.gd_screen.show_loading()
        QApplication.processEvents()  # ローディング表示を確実に実行
        
        # 少し遅延させてからシグナルを発火（画面遷移とローディング表示を確実にするため）
        QTimer.singleShot(100, lambda: self.start_gd_requested.emit(self.current_username, self.current_gd_round))
    
    def _on_gd_start_cancelled(self):
        """GD開始キャンセル → 適切な画面へ戻る"""
        if self.current_gd_round == 1:
            # 1回目: ユーザー名入力画面へ戻る
            self.stacked_widget.setCurrentIndex(2)  # ユーザー名入力画面へ戻る
        else:
            # 2回目: フィードバック画面または統制群用画面へ戻る
            if self.experiment_group == "experimental":
                self.stacked_widget.setCurrentIndex(5)  # フィードバック画面へ
            else:
                self.stacked_widget.setCurrentIndex(6)  # 統制群用画面へ
    
    def _on_next_gd_requested(self):
        """2回目GD開始要求 → 2回目GD開始確認画面へ"""
        self.current_gd_round = 2
        # 2回目GD開始確認画面を表示
        # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 2分
        if TEST_MODE:
            thinking_time_text = "1秒間"
            thinking_seconds = 1
        elif DEV_MODE:
            thinking_time_text = "10秒間"
            thinking_seconds = 10
        else:
            thinking_time_text = "2分間"
            thinking_seconds = 120
        self.gd_start_confirm_screen.set_message(
            f"これから2回目のグループディスカッションを開始します。\n手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        )
        self.stacked_widget.setCurrentIndex(3)  # GD開始確認画面へ
        # システムアナウンス: 画面表示時にアナウンス
        announcement = f"これから2回目のグループディスカッションを開始します。手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        # アナウンスは非同期で実行（UIをブロックしない）
        from threading import Thread
        from PySide6.QtCore import QTimer
        thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
        thread.start()
        # アナウンス再生後に思考時間タイマーを開始（テキストの長さから再生時間を推定）
        # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
        estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
        QTimer.singleShot(int(estimated_duration * 1000), lambda: self.gd_start_confirm_screen.start_thinking_time(thinking_seconds))
    
    def _on_thinking_timeout(self):
        """思考時間終了時のアナウンス（表示後に再生）"""
        # 表示後にアナウンスを再生（少し遅延させて表示を確実にする）
        from PySide6.QtCore import QTimer
        from threading import Thread
        announcement = "思考時間が終了しました。「開始する」ボタンを押してください。"
        # アナウンスを再生
        QTimer.singleShot(500, lambda: Thread(target=lambda: _announce_system_message(announcement), daemon=True).start())
        # アナウンス再生時間を推定して、その後にボタンを有効化
        # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ
        estimated_duration = len(announcement) * 0.1 + 1.0
        QTimer.singleShot(int((500 + estimated_duration * 1000)), lambda: self.gd_start_confirm_screen.enable_confirm_button_after_announcement())
    
    def _on_reading_timeout_feedback(self):
        """フィードバック読書時間終了時 → 2回目GD開始確認画面へ"""
        self.current_gd_round = 2
        # 2回目GD開始確認画面を表示
        # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 2分
        if TEST_MODE:
            thinking_time_text = "1秒間"
            thinking_seconds = 1
        elif DEV_MODE:
            thinking_time_text = "10秒間"
            thinking_seconds = 10
        else:
            thinking_time_text = "2分間"
            thinking_seconds = 120
        self.gd_start_confirm_screen.set_message(
            f"これから2回目のグループディスカッションを開始します。\n手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        )
        self.stacked_widget.setCurrentIndex(3)  # GD開始確認画面へ
        # システムアナウンス: 画面表示時にアナウンス
        announcement = f"これから2回目のグループディスカッションを開始します。手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        # アナウンスは非同期で実行（UIをブロックしない）
        from threading import Thread
        from PySide6.QtCore import QTimer
        thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
        thread.start()
        # アナウンス再生後に思考時間タイマーを開始（テキストの長さから再生時間を推定）
        # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
        estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
        QTimer.singleShot(int(estimated_duration * 1000), lambda: self.gd_start_confirm_screen.start_thinking_time(thinking_seconds))
    
    def _on_reading_timeout_control(self):
        """統制群読書時間終了時 → 2回目GD開始確認画面へ"""
        self.current_gd_round = 2
        # 2回目GD開始確認画面を表示
        # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 2分
        if TEST_MODE:
            thinking_time_text = "1秒間"
            thinking_seconds = 1
        elif DEV_MODE:
            thinking_time_text = "10秒間"
            thinking_seconds = 10
        else:
            thinking_time_text = "2分間"
            thinking_seconds = 120
        self.gd_start_confirm_screen.set_message(
            f"これから2回目のグループディスカッションを開始します。\n手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        )
        self.stacked_widget.setCurrentIndex(3)  # GD開始確認画面へ
        # システムアナウンス: 画面表示時にアナウンス
        announcement = f"これから2回目のグループディスカッションを開始します。手元の「②」と書かれている用紙を裏返して、記載されているテーマについて{thinking_time_text}考えてください。"
        # アナウンスは非同期で実行（UIをブロックしない）
        from threading import Thread
        from PySide6.QtCore import QTimer
        thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
        thread.start()
        # アナウンス再生後に思考時間タイマーを開始（テキストの長さから再生時間を推定）
        # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
        estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
        QTimer.singleShot(int(estimated_duration * 1000), lambda: self.gd_start_confirm_screen.start_thinking_time(thinking_seconds))
    
    # 外部から呼ばれるメソッド
    def set_theme(self, theme):
        """テーマを設定"""
        self.current_theme = theme
    
    def set_minutes(self, minutes_text):
        """議事録を更新"""
        self.gd_screen.update_minutes(minutes_text)
    
    def update_speaker(self, speaker_name):
        """発言者を更新"""
        self.gd_screen.update_speaker(speaker_name)
    
    def update_timer(self, remaining_minutes, remaining_seconds):
        """残り時間を更新"""
        self.gd_screen.update_timer(remaining_minutes, remaining_seconds)
    
    def set_feedback(self, feedback_dict):
        """フィードバックを表示（1回目終了時）"""
        self.first_gd_feedback = feedback_dict
        
        # 実験群・統制群共通: CSVデータを保存（統制群はCSVのみ、実験群はCSV+Markdown）
        if hasattr(self, 'manager') and self.manager:
            try:
                filepath = self.manager.save_feedback_report(
                    feedback_dict, 
                    self.current_fullname,  # フルネームで保存
                    round_number=1,
                    experiment_group=self.experiment_group
                )
                if self.experiment_group == "experimental":
                    print(f"[システム]: フィードバックを自動保存しました: {filepath}")
                else:
                    print(f"[システム]: CSVデータを自動保存しました: {filepath}")
            except Exception as e:
                print(f"[警告]: データの自動保存に失敗しました: {e}")
        
        if self.experiment_group == "experimental":
            # 実験群: フィードバック画面を表示
            self.feedback_screen.set_feedback(feedback_dict)
            self.stacked_widget.setCurrentIndex(5)  # フィードバック画面へ（index 5）
            # システムアナウンス: 読書時間開始
            # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 5分
            if TEST_MODE:
                reading_time_text = "1秒間"
                reading_seconds = 1
            elif DEV_MODE:
                reading_time_text = "10秒間"
                reading_seconds = 10
            else:
                reading_time_text = "5分間"
                reading_seconds = 300
            announcement = f"AIからのフィードバックレポートを{reading_time_text}読み、2回目のグループディスカッションに備えてください。"
            # アナウンスは非同期で実行（UIをブロックしない）
            from threading import Thread
            from PySide6.QtCore import QTimer
            thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
            thread.start()
            # アナウンス再生後に読書時間タイマーを開始（テキストの長さから再生時間を推定+3秒）
            # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
            estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
            QTimer.singleShot(int(estimated_duration * 1000), lambda: self.feedback_screen.start_reading_time(reading_seconds))
        else:
            # 統制群: 学習用ドキュメント画面へ（CSVは既に保存済み）
            self.stacked_widget.setCurrentIndex(6)  # 統制群用画面へ（index 6）
            # システムアナウンス: 読書時間開始
            # テストモード: 1秒（実質スキップ）、開発モード: 10秒、本番: 5分
            if TEST_MODE:
                reading_time_text = "1秒間"
                reading_seconds = 1
            elif DEV_MODE:
                reading_time_text = "10秒間"
                reading_seconds = 10
            else:
                reading_time_text = "5分間"
                reading_seconds = 300
            announcement = f"ファシリテーションに関するハンドブックを{reading_time_text}読み、2回目のグループディスカッションに備えてください。"
            # アナウンスは非同期で実行（UIをブロックしない）
            from threading import Thread
            from PySide6.QtCore import QTimer
            thread = Thread(target=lambda: _announce_system_message(announcement), daemon=True)
            thread.start()
            # アナウンス再生後に読書時間タイマーを開始（テキストの長さから再生時間を推定+3秒）
            # speaking_rate=1.2の場合、1文字あたり約0.083秒、安全のため文字数*0.1秒+1秒のバッファ+3秒追加
            estimated_duration = len(announcement) * 0.1 + 1.0 + 3.0
            QTimer.singleShot(int(estimated_duration * 1000), lambda: self.control_after_first_screen.start_reading_time(reading_seconds))
    
    def _on_feedback_progress(self, message):
        """フィードバック生成進捗を表示"""
        # フィードバック画面に切り替えて進捗を表示
        self.stacked_widget.setCurrentIndex(5)  # フィードバック画面へ（index 5）
        self.feedback_screen.show_progress(message)
    
    def set_manager(self, manager):
        """GDManagerへの参照を保持"""
        self.manager = manager
    
    def show_system_speaking(self):
        """システム発言中バナーを表示"""
        self.gd_screen.show_system_speaking()
    
    def hide_system_speaking(self):
        """システム発言中バナーを非表示"""
        self.gd_screen.hide_system_speaking()
    
    def _on_second_gd_finished(self, feedback_dict):
        """2回目GD終了時の処理"""
        # 自動保存（2回目）- アナウンスはGDThread側で既に再生済み
        if hasattr(self, 'manager') and self.manager:
            try:
                filepath = self.manager.save_feedback_report(
                    feedback_dict, 
                    self.current_fullname,  # フルネームで保存
                    round_number=2,
                    experiment_group=self.experiment_group
                )
                print(f"[システム]: 2回目のフィードバックを自動保存しました: {filepath}")
                # スコア保存後にポップアップを表示
                from PySide6.QtCore import QTimer
                QTimer.singleShot(500, lambda: self._show_data_saved_popup())
            except Exception as e:
                print(f"[警告]: 2回目のフィードバックの自動保存に失敗しました: {e}")
                # エラー時もポップアップを表示
                from PySide6.QtCore import QTimer
                QTimer.singleShot(500, lambda: self._show_data_saved_popup())
    
    def _show_data_saved_popup(self):
        """データ保存完了ポップアップを表示"""
        msg = QMessageBox(self)
        msg.setWindowTitle("データ保存")
        msg.setText("データが保存されました。")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
        
        # ポップアップを閉じたらシステム終了
        QApplication.instance().quit()
    
    def _show_finish_popup(self):
        """実験終了ポップアップを表示"""
        msg = QMessageBox(self)
        msg.setWindowTitle("実験終了")
        msg.setText("実験お疲れさまでした。管理者を読んでください。")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
        
        # ポップアップを閉じたらシステム終了
        QApplication.instance().quit()

# --- GDManagerをスレッドで実行するためのクラス ---
class GDThread(QThread):
    finished = pyqtSignal(dict) # GD終了時にフィードバックを送信するシグナル
    minutes_updated = pyqtSignal(str)  # リアルタイム議事録更新用シグナル
    speaker_changed = pyqtSignal(str)  # 発言者変更用シグナル
    timer_updated = pyqtSignal(int, int)  # 残り時間更新用シグナル (分, 秒)
    feedback_progress = pyqtSignal(str)  # フィードバック生成進捗用シグナル
    role_updated = pyqtSignal(str, str)  # 役割更新用シグナル (参加者名, 役割)
    system_speaking = pyqtSignal(bool)  # システム発話中フラグ (True=開始, False=終了)
    
    def __init__(self, manager, round_number=1, experiment_group=None):
        super().__init__()
        self.manager = manager
        self.round_number = round_number  # ラウンド番号を保持
        self.experiment_group = experiment_group  # 実験群/統制群
        self._running = True  # スレッド停止フラグ
        self._gd_started = False  # GD開始フラグ
        self._timer_started = False  # タイマー開始フラグ
        self._timer_thread = None  # タイマースレッドの参照
    
    def stop(self):
        """外部からスレッドの停止を要求する"""
        self._running = False
    
    def start_gd(self):
        """外部からGD開始を要求する"""
        self._gd_started = True
    
    def start_timer(self):
        """外部からタイマー開始を要求する（キックオフメッセージ終了後に呼ばれる）"""
        if not self._timer_started:
            # タイマー開始時にstart_timeを更新（キックオフメッセージ終了時点から計測）
            self.manager.start_time = time.time()
            self._timer_started = True
            print("[システム]: タイマーを開始しました")
    
    def run(self):
        # GD開始まで待機
        while self._running and not self._gd_started:
            self.msleep(100)  # 100ms待機
        
        if not self._running:
            return
            
        # 注意: _initialize_gd()はメインスレッド側（on_start_gd_with_username_and_round）で
        # 既に実行されているため、ここでは呼ばない（重複を防ぐ）
        
        # タイマースレッドを開始（待機状態）
        self._timer_thread = threading.Thread(target=self._update_timer_loop, daemon=True)
        self._timer_thread.start()
        
        # 議事録更新は発言終了時に自動的に行われるため、定期的な更新スレッドは不要
        # GDManagerにgd_threadへの参照を設定
        self.manager.gd_thread = self
        
        # 会話ループを実行
        try:
            running_gd = self.manager.run_conversation_loop(self.speaker_changed, gd_thread=self)
        except KeyboardInterrupt:
            # Ctrl+C が来た場合はループを抜ける
            running_gd = False
        except Exception as e:
            print(f"GDThreadエラー: {e}")
            running_gd = False
        
        # GD終了後にシステムメッセージを音声で通知
        if self.round_number == 1:
            # 1回目: フィードバックを生成（進捗表示付き）
            if self.experiment_group == "experimental":
                end_message = "グループディスカッションを終了します。お疲れ様でした。フィードバックレポートを生成します。"
            else:
                end_message = "グループディスカッションを終了します。お疲れ様でした。"
            # システムアナウンスを再生（UIに反映されるようにgd_threadを渡す）
            # GD終了後すぐに音声を再生するため、待機時間を削除
            self.manager._synthesize_and_play_system_message(end_message, gd_thread=self)
            
            # フィードバックを生成し、シグナルで送信（進捗表示付き、統制群の場合はAIフィードバックは生成しない）
            report = self.manager.generate_simple_feedback_report(
                progress_signal=self.feedback_progress if self.experiment_group == "experimental" else None,
                experiment_group=self.experiment_group
            )
            self.finished.emit(report)
        else:
            # 2回目: フィードバックを生成するが表示はしない（データは自動保存される）
            end_message = "グループディスカッションが終了しました。実験お疲れさまでした。管理者を呼んでください。"
            self.manager._synthesize_and_play_system_message(end_message, gd_thread=self)
            # フィードバックを生成（進捗表示なし、表示もされない、統制群の場合はAIフィードバックは生成しない）
            report = self.manager.generate_simple_feedback_report(experiment_group=self.experiment_group)
            self.finished.emit(report)
    
    def _update_timer_loop(self):
        """残り時間を1秒ごとに更新するループ"""
        # タイマー開始を待機
        while self._running and not self._timer_started:
            time.sleep(0.1)
        
        if not self._running:
            return
        
        # タイマー開始時点のstart_timeを記録（正確な計測のため）
        # start_timer()でstart_timeが更新された時点から計測する
        timer_start_time = self.manager.start_time
        
        while self._running:
            try:
                elapsed_seconds = int(time.time() - timer_start_time)
                remaining_seconds = max(0, self.manager.time_limit_minutes * 60 - elapsed_seconds)
                remaining_minutes = remaining_seconds // 60
                remaining_secs = remaining_seconds % 60
                
                # シグナルで残り時間を通知
                self.timer_updated.emit(remaining_minutes, remaining_secs)
                
                # 1秒待機
                time.sleep(1)
                
                # 時間切れの場合はループを抜ける
                if remaining_seconds <= 0:
                    break
            except Exception as e:
                print(f"タイマー更新エラー: {e}")
                break

class GDManager:
    """
    グループディスカッションの進行を管理する中央オーケストレータークラス。
    Gemini APIを利用してユーザーとのGDシミュレーションを管理する。
    """
    # コンストラクタにusername引数を追加
    def __init__(self, gui_window, username="ユーザー", gd_theme=None, num_ai_participants=3): 
        print("GDManagerを初期化中...")
        
        self.gui_window = gui_window # GUIインスタンスを保持
        self.username = username  # ユーザー名を保持
        # テーマが渡されていればそれを使用、なければエラー（固定テーマを使用するため）
        if gd_theme is None:
            raise ValueError("gd_themeは必須です。get_fixed_gd_theme(round_number)を使用してください。")
        self.gd_theme = gd_theme
        self.num_ai_participants = num_ai_participants
        self.conversation_history = []
        # テストモード: 2分、開発モード: 10秒、本番: 10分
        if TEST_MODE:
            self.time_limit_minutes = 2
        elif DEV_MODE:
            self.time_limit_minutes = 10 / 60
        else:
            self.time_limit_minutes = 10
        self.start_time = time.time() 
        self.current_speaker = "システム" 
        self.roles_assigned = False
        self.kickoff_announced = False  # 本格開始アナウンス済みフラグ
        self.first_speech_done = False  # 最初の発言が完了したかどうか（タイムアウト制御用）
        self.conversation_active = True  # 会話がアクティブかどうか
        self.gd_thread = None  # GDThreadへの参照（議事録更新用）
        
        # --- APIクライアントの初期化 ---
        try:
            # self.speech_client = speech.SpeechClient()  # ASRクライアント
            # self.tts_client = texttospeech.TextToSpeechClient() # TTSクライアント
            
            # --- Gemini APIの認証とクライアント初期化 ---
            # .envファイルから GOOGLE_API_KEY 環境変数を読み込みます
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # <-- GOOGLE_API_KEY を .env に設定すること
            
            # 安全性設定（教育用途に適したレベル: 高レベルの有害コンテンツのみブロック）
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
            ]
            
            # GD中・採点用モデル（高速）
            self.gemini_model = genai.GenerativeModel(
                GEMINI_MODEL,
                safety_settings=self.safety_settings
            )
            # フィードバック生成専用モデル（高精度）
            try:
                self.gemini_feedback_model = genai.GenerativeModel(
                    GEMINI_FEEDBACK_MODEL,
                    safety_settings=self.safety_settings
                )
            except Exception as e:
                # フィードバック専用モデルの初期化に失敗した場合は、通常モデルで代用
                print(f"フィードバック専用モデル({GEMINI_FEEDBACK_MODEL})の初期化に失敗しました。通常モデルで代用します: {e}")
                self.gemini_feedback_model = self.gemini_model
            print("全APIクライアントを GDManager 内で初期化しました。")
        except Exception as e:
            error_msg = f"エラー: APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}\nヒント: GOOGLE_APPLICATION_CREDENTIALS と GOOGLE_API_KEY が正しく設定されていますか？"
            print(error_msg)
            raise RuntimeError(error_msg)  # exit()の代わりに例外を投げる
        
        # PyAudioインスタンスを再利用するため、クラス変数として保持
        self.p_audio = None  # 遅延初期化（最初の使用時に作成）
        
        # --- 参加者とペルソナの設定 ---
        # 日本語の一般的な名前の候補リストを作成
        name_candidates = [
            "田中", "佐藤", "鈴木", "高橋", "渡辺", "伊藤", "山本", "中村", "小林", "加藤",
            "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
            "山崎", "森", "池田", "橋本", "阿部", "石川", "中島", "小野", "藤井", "原田",
            "岡田", "後藤", "長谷川", "村上", "近藤", "前田", "石田", "坂本", "遠藤", "青木"
            ] 
        # 候補からランダムに4名を選出（AI参加者用）
        selected_ai_names = random.sample(name_candidates, self.num_ai_participants)
        
        # 参加者辞書を構築（ユーザーは入力された名字を使用）
        self.participants = {
            self.username: {"role": "ユーザー", "persona": "あなたはGDを円滑に進行し、結論に導く責任があるファシリテーターです。"}
        }
        self.ai_voice_map = {}
        
        # AI参加者の名前とタイプのマッピングを保存
        self.ai_name_to_type = {}
        
        # より識別しやすい声色とパラメータの設定（3人用）
        # Neural2-B: 女性, Neural2-C: 男性, Neural2-D: 男性
        voice_configs = [
            {"voice": "ja-JP-Neural2-C", "pitch": -2.0, "rate": 1.0},   # AI1: 低音男性、ゆっくり
            {"voice": "ja-JP-Neural2-B", "pitch": +2.0, "rate": 1.3},   # AI2: 高音女性、早口
            {"voice": "ja-JP-Neural2-D", "pitch": 0.0, "rate": 1.1},    # AI3: 標準男性、少し早め
        ]
        
        self.ai_voice_config = {}  # 新しい音声設定マッピング
        
        # 3つのペルソナタイプを順番に割り当て（積極派・慎重派・消極派）
        persona_types = ["積極派", "慎重派", "消極派"]
        for i, ai_id in enumerate(selected_ai_names):
            # ペルソナタイプを順番に割り当て（3人以上の場合も循環）
            persona_type = persona_types[i % len(persona_types)]
            persona_text = self._get_default_ai_persona(ai_id, persona_type)
            # 積極性レベルを設定（発話タイミング制御用）
            # 積極派: 0.8, 慎重派: 0.5, 消極派: 0.35
            activity_level = {"積極派": 0.8, "慎重派": 0.5, "消極派": 0.35}[persona_type]
            self.participants[ai_id] = {
                "role": ai_id, 
                "persona": persona_text,
                "persona_type": persona_type,
                "activity_level": activity_level  # 発話タイミング制御用
            }
            
            # AI参加者の名前→タイプマッピング
            self.ai_name_to_type[ai_id] = f"ai_{i+1}"
            
            # AI参加者ごとに異なる声色とパラメータを割り当てる
            config = voice_configs[i] if i < len(voice_configs) else voice_configs[0]
            self.ai_voice_config[ai_id] = config
            # 後方互換性のため ai_voice_map も維持
            self.ai_voice_map[ai_id] = config["voice"]
        
        print("GDManagerの初期化が完了しました。")
        print(f"参加者: {self.username}(ユーザー), {', '.join(selected_ai_names)}(AI)")
        print("音声設定:")
        for ai_id in selected_ai_names:
            config = self.ai_voice_config[ai_id]
            print(f"  {ai_id}: {config['voice']} (pitch={config['pitch']}, rate={config['rate']})")
        
        # GUIに参加者情報を設定
        participant_names = {self.username: "user"}
        for ai_name in selected_ai_names:
            participant_names[ai_name] = self.ai_name_to_type[ai_name]
        self.gui_window.gd_screen.set_participants(participant_names)
        
        # 初期メッセージの同期再生はGUI表示をブロックするため、
        # コンストラクタ内では呼ばない（外部から非同期に開始する）。
        # self._initialize_gd() は別スレッドで起動してください。
        # self._initialize_gd() 
        
    def _get_default_ai_persona(self, ai_id, persona_type):
        """
        AI参加者ごとの詳細なペルソナを設定する（外崎・伊藤の研究に基づく）
        
        Args:
            ai_id: AI参加者の名前
            persona_type: "積極派", "慎重派", "消極派" のいずれか
        
        Returns:
            str: 詳細なペルソナ定義
        """
        if persona_type == "積極派":
            return f"""あなたはGDの参加者である{ai_id}です。以下の特徴を持っています：

【性格・行動パターン】
- 誰よりも先に発言する傾向がある
- 思いついたらすぐに言う
- 話が長くなりがち
- 積極的にアイデアを提案する
- 新しい視点を提供することを重視する

【発言スタイル】
- 自発的に発言する
- 発言の長さは中〜長め
- 具体的な提案を重視する
- 「私からは〜」「私の意見としては〜」といった表現を使う

【注意事項】
- 文脈（現在のフェーズや議論の流れ）を理解して発言すること
- 他の参加者の発言を踏まえて、建設的に議論を進めること"""
        
        elif persona_type == "慎重派":
            return f"""あなたはGDの参加者である{ai_id}です。以下の特徴を持っています：

【性格・行動パターン】
- 積極派の意見に対して実現可能性を検討する視点を提供する
- 実現可能性を重視し、具体的な方法を考える
- 建設的な視点から問題点を検討する
- 議論を深掘りすることを好む
- 協調的に議論を進めることを重視する

【発言スタイル】
- 積極派の後に発言することが多い
- 発言の長さは中程度
- 「確かにそうですね。実現するために〜を考慮すると良いかもしれません」「〜という点も検討すると良いと思います」といった建設的な表現を使う
- 慎重に検討する姿勢を示すが、否定的にならない

【注意事項】
- 文脈（現在のフェーズや議論の流れ）を理解して発言すること
- 否定的にならず、建設的で協調的な発言を心がけること
- 批判的になりすぎず、議論を前向きに進めることを重視すること"""
        
        else:  # 消極派
            return f"""あなたはGDの参加者である{ai_id}です。以下の特徴を持っています：

【性格・行動パターン】
- 基本的には指名されたら答えるが、時々自発的に発言することもある
- 指名されたら短く答える
- 「特にないです」「そうですね」と言いがち
- 発言が短い
- 他の参加者の意見に同調することが多い

【発言スタイル】
- 時々自発的に発言することもあるが、基本的には指名された時に発言する傾向がある
- 発言の長さは短め
- 「そうですね」「確かに」「特にないです」といった短い応答が多い
- 積極派ほど頻繁ではないが、適切なタイミングでは自発的に発言することもある

【注意事項】
- 文脈（現在のフェーズや議論の流れ）を理解して発言すること
- 完全に沈黙するのではなく、適切なタイミングでは自発的に発言することも心がけること"""
    
    def _initialize_gd(self):
        """GD開始時の初期メッセージを発言させる
        
        ※GUIの更新（テーマ表示や役職表示）はメインスレッド側
          （on_start_gd_with_username_and_round）で行う。
        """
        # 音声で読み上げるのはタイトル部分のみ（1行目）にする
        theme_title = self.gd_theme.splitlines()[0] if self.gd_theme else ""
        initial_message = (
            f"グループディスカッションを始めます。本日のテーマは『{theme_title}』です。"
            f"まず、{self.username}さんから順番に、お名前と一言だけ簡単に自己紹介をお願いします。"
        )
        self.conversation_history.append({"speaker": "システム", "content": initial_message})
        print(f"[システム]: {initial_message}")
        self._synthesize_and_play_system_message(initial_message) 
        self.current_speaker = "システム"
    
    def _generate_ai_prompt(self, ai_id, current_task_for_ai, include_current_history=True):
        """
        特定のAI参加者に対するLLMへのプロンプト（指示文）を構築する。
        Gemini APIの形式に合わせる。
        """
        # Geminiはロールを'user'/'model'で区別することが多いため、調整
        messages_content = []
        
        # システム指示
        persona_info = self.participants[ai_id]
        
        system_instruction_content = (
            f"あなたはグループディスカッションの参加者である{ai_id}です。\n\n"
            f"【あなたのペルソナ】\n"
            f"{persona_info['persona']}\n\n"
            f"【重要な指示】\n"
            f"- これまでの会話履歴をよく読み、文脈を理解して発言してください\n"
            f"- 直前の発言（特に他の参加者の発言）を踏まえて、自然な会話の流れを作ってください\n"
            f"- ペルソナに忠実に、人間らしい自然な発言を心がけてください\n"
            f"- 「です・ます」調で話してください\n"
            f"- 発言は2〜3文程度の簡潔な長さにしてください（長すぎる発言は避けてください）\n"
            f"- 会話の流れに沿って、適度に感情や反応を示してください\n"
            f"- ビジネスや学術的な議論の文脈で、適切で建設的な発言をしてください\n"
            f"- 役割分担の自己紹介は既に完了しているため、再度「{ai_id}です、よろしくお願いします」や「{ai_id}が{self.participants[ai_id].get('assigned_role', '役割')}を担当します」などの自己紹介や役割の再確認は行わないでください\n"
            f"- 常に適切で建設的な内容のみを発言し、不適切な表現や内容は避けてください\n"
            f"- 質問がある場合は、具体的に答えてください\n"
        )
        messages_content.append({"role": "user", "parts": [system_instruction_content]})
        messages_content.append({"role": "model", "parts": ["了解しました。"]}) # AIの初期応答をシミュレート
        
        # 過去の会話履歴（直近の会話を優先的に含める）
        if include_current_history:
            # 直近15ターン分の会話履歴を含める（文脈理解のため、10→15に増加）
            # ただし、会話履歴が長い場合は要約を含める
            recent_history = self.conversation_history[-15:] if len(self.conversation_history) > 15 else self.conversation_history
            
            # 会話履歴が長い場合（15ターン以上）、古い部分をLLMで要約
            if len(self.conversation_history) > 15:
                # 古い会話履歴（15ターンより前）を要約
                older_history = self.conversation_history[:-15]
                if older_history:
                    # 古い会話履歴をテキストに変換
                    older_history_text = "\n".join([
                        f"[{msg['speaker']}]: {msg['content']}" 
                        for msg in older_history 
                        if msg['speaker'] != "システム"
                    ])
                    
                    # LLMで要約を生成
                    summary_prompt = f"""以下のグループディスカッションの会話履歴を要約してください。

【会話履歴】
{older_history_text}

【指示】
- 重要な決定事項、主要な意見、役割分担の情報を簡潔にまとめてください
- 3〜5文程度の簡潔な要約にしてください
- 会話の流れと文脈を保持してください

要約:"""
                    
                    try:
                        summary_messages = [
                            {"role": "user", "parts": [summary_prompt]}
                        ]
                        summary_response = self.gemini_model.generate_content(summary_messages)
                        if summary_response._result.candidates:
                            summary_text = summary_response.text.strip()
                            messages_content.append({"role": "user", "parts": [f"【これまでの会話の要約】\n{summary_text}"]})
                            messages_content.append({"role": "model", "parts": ["了解しました。"]})
                        else:
                            # LLM要約に失敗した場合は簡易版を使用
                            messages_content.append({"role": "user", "parts": [f"（これまでの会話: {len(older_history)}件の発言がありました）"]})
                            messages_content.append({"role": "model", "parts": ["了解しました。"]})
                    except Exception as e:
                        print(f"会話履歴要約エラー: {e}")
                        # エラー時は簡易版を使用
                        messages_content.append({"role": "user", "parts": [f"（これまでの会話: {len(older_history)}件の発言がありました）"]})
                        messages_content.append({"role": "model", "parts": ["了解しました。"]})
            
            # 直近の会話履歴を追加
            for msg in recent_history:
                if msg['speaker'] == "システム": continue # システムメッセージはLLMに渡さない
                # Geminiでは'user'と'model'が交互にくる必要がある
                role_gemini = "user" if msg['speaker'] == self.username else "model"
                messages_content.append({"role": role_gemini, "parts": [msg['content']]})
        
        # 最新のタスク指示（文脈を考慮した指示）
        remaining_minutes = int((self.time_limit_minutes * 60 - (time.time() - self.start_time)) / 60)
        
        # タスク指示が空の場合は、会話の流れから自発的に判断するように指示
        if not current_task_for_ai or current_task_for_ai.strip() == "":
            task_instruction = (
                f"会話履歴と文脈を理解して、ペルソナに忠実に、自然に応答してください。\n"
                f"あなたの役割（{self.participants[ai_id].get('assigned_role', 'なし')}）を意識して、適切に発言してください。"
            )
        else:
            task_instruction = current_task_for_ai
        
        # 参加者情報と役割を取得
        all_participants_info = []
        for participant_name, participant_data in self.participants.items():
            role = participant_data.get('assigned_role', 'なし')
            if role != 'なし':
                all_participants_info.append(f"{participant_name}（{role}）")
            else:
                all_participants_info.append(participant_name)
        
        # 書記がまだ設定されておらず、ユーザーが役割分担について言及している場合のみ追加指示
        role_assignment_note = ""
        if not self._has_recorder() and self._user_mentioned_role_assignment():
            role_assignment_note = (
                f"- 会話の中で役割分担（書記やタイムキーパーを決める）が促された場合は、"
                f"自然な流れで「私が書記を担当します」「書記をお願いします」など、書記を担当する旨を発言してください。"
                f"書記を担当すると宣言する際は、議事録を作成して共有する旨も併せて発言してください。\n"
            )
        
        messages_content.append({"role": "user", "parts": [
            f"【GDの状況】\n"
            f"テーマ: {self.gd_theme.splitlines()[0] if self.gd_theme else '未設定'}\n"
            f"制限時間: 10分\n"
            f"参加者: {', '.join(all_participants_info)}\n"
            f"あなたの役割: {self.participants[ai_id].get('assigned_role', 'なし')}\n\n"
            f"【あなたへの指示】\n"
            f"{task_instruction}\n\n"
            f"【重要な注意事項】\n"
            f"- 発言は2〜3文程度の簡潔な長さにしてください（長すぎる発言は避けてください）\n"
            f"{role_assignment_note}"
            f"- 役割分担の自己紹介は既に完了しているため、再度「{ai_id}です、よろしくお願いします」や「{ai_id}が{self.participants[ai_id].get('assigned_role', '役割')}を担当します」などの自己紹介や役割の再確認は行わないでください\n\n"
            f"上記の指示に従い、ペルソナに忠実に、自然な日本語で発言を生成してください。"
        ]})
    
        return messages_content
    
    def _get_ai_response_streaming(self, ai_id, task_for_ai, include_current_history=True, max_retries=2):
        """
        特定のAI参加者（ai_id）の発言を非ストリーミングモードで生成し、
        完全な応答を取得してから文単位でyieldする（TTS並列実行用）
        
        Args:
            ai_id: AI参加者のID
            task_for_ai: タスク指示
            include_current_history: 会話履歴を含めるか
            max_retries: 最大リトライ回数
        
        Yields:
            str: 生成された文（句点で区切られた単位）
        """
        messages = self._generate_ai_prompt(ai_id, task_for_ai, include_current_history)
        print(f" (GDマネージャー -> Geminiへの指示 for {ai_id}): {task_for_ai[:80]}...")

        # 非ストリーミングモードで試行（ストリーミングモードは使用しない）
        for retry_count in range(max_retries + 1):
            try:
                # 非ストリーミングモードで試行
                response = self.gemini_model.generate_content(messages, stream=False)
                
                if not response._result.candidates:
                    print(f"[警告] 非ストリーミングモードで応答がブロックされました。プロンプトを調整して再試行します。")
                    # プロンプトをより安全な内容に調整して再試行
                    if retry_count < max_retries:
                        # プロンプトを簡潔にして再試行
                        simplified_messages = self._generate_ai_prompt(ai_id, "会話履歴と文脈を理解して、ペルソナに忠実に自然に応答してください。", include_current_history)
                        messages = simplified_messages
                        continue
                    else:
                        # 最大リトライ回数に達した場合のみフォールバック
                        fallback_message = self._generate_contextual_fallback(ai_id, task_for_ai)
                        yield fallback_message
                        return
                
                response_text = response.text.strip()
                if response_text:
                    # 完全な応答を取得してから、文単位で分割してyield
                    # 句点（。！？）で分割し、完全な文のみを返す
                    import re
                    # 句点で分割（句点を含む）
                    sentences = re.split(r'([。！？])', response_text)
                    # 句点と文を組み合わせて完全な文を作成
                    complete_sentences = []
                    for i in range(0, len(sentences) - 1, 2):
                        if i + 1 < len(sentences):
                            sentence = (sentences[i] + sentences[i + 1]).strip()
                            if sentence:
                                complete_sentences.append(sentence)
                    # 最後の文（句点がない場合）も追加
                    if len(sentences) % 2 == 1 and sentences[-1].strip():
                        complete_sentences.append(sentences[-1].strip())
                    
                    # 完全な文のみをyield（途中で終わらないように）
                    for sentence in complete_sentences:
                        if sentence:
                            yield sentence
                    return  # 成功したので終了
                else:
                    # 応答が空の場合、プロンプトを調整して再試行
                    if retry_count < max_retries:
                        simplified_messages = self._generate_ai_prompt(ai_id, "会話履歴と文脈を理解して、ペルソナに忠実に自然に応答してください。", include_current_history)
                        messages = simplified_messages
                        continue
                    else:
                        # 最大リトライ回数に達した場合のみフォールバック
                        fallback_message = self._generate_contextual_fallback(ai_id, task_for_ai)
                        yield fallback_message
                        return
                            
            except Exception as e:
                print(f"[エラー] Gemini呼び出し中にエラーが発生しました (リトライ {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    # リトライを試みる
                    import time
                    time.sleep(0.5)  # 少し待ってからリトライ
                    continue
                else:
                    # 最大リトライ回数に達した場合のみフォールバック
                    fallback_message = self._generate_contextual_fallback(ai_id, task_for_ai)
                    yield fallback_message
                    return
        
        # すべてのリトライが失敗した場合（通常はここには到達しない）
        fallback_message = self._generate_contextual_fallback(ai_id, task_for_ai)
        yield fallback_message
    
    def _generate_contextual_fallback(self, ai_id, task_for_ai):
        """
        文脈に応じたフォールバックメッセージを生成する
        
        Args:
            ai_id: AI参加者のID
            task_for_ai: 現在のタスク指示
        
        Returns:
            str: 文脈に応じたフォールバックメッセージ
        """
        # 役割分担に関するタスクの場合
        if "役割分担" in task_for_ai or "役割" in task_for_ai:
            assigned_role = self.participants[ai_id].get('assigned_role', '')
            if assigned_role:
                return f"{assigned_role}を担当します。"
            else:
                return "了解しました。"
        
        # その他の場合は、会話の流れに沿った簡潔な応答
        return "了解しました。"
    
    def _get_ai_response(self, ai_id, task_for_ai, include_current_history=True):
        """
        特定のAI参加者（ai_id）の発言をLLM（Gemini）に生成させる。
        返り値は常に文字列（空のときはフォールバックテキスト）にする。
        ストリーミング版を使用して全文を結合。
        """
        # 考えている状態を表示
        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
            self.gui_window.gd_screen.show_ai_thinking(ai_id)
        
        try:
            full_response = ""
            for sentence in self._get_ai_response_streaming(ai_id, task_for_ai, include_current_history):
                full_response += sentence
            return full_response if full_response else f"（{ai_id}）応答が空です。"
        finally:
            # 考えている状態を非表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.hide_ai_status()

    def _clean_text_for_tts(self, text):
        """
        TTSで読み上げる前にマークダウン記号や特殊文字を除去する
        """
        import re
        
        # マークダウンの強調記号を除去（**, *, __, _）
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **太字** → 太字
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *斜体* → 斜体
        text = re.sub(r'__([^_]+)__', r'\1', text)       # __太字__ → 太字
        text = re.sub(r'_([^_]+)_', r'\1', text)         # _斜体_ → 斜体
        
        # その他のマークダウン記号を除去
        text = re.sub(r'#+\s*', '', text)                # 見出し記号 (#, ##)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # リンク [text](url) → text
        text = re.sub(r'`([^`]+)`', r'\1', text)         # コード `code` → code
        
        return text

    def _synthesize_tts(self, text, ai_id):
        """テキストを音声合成（再生なし）- 個別のpitch/rate設定対応"""
        # マークダウン記号を除去
        cleaned_text = self._clean_text_for_tts(text)
        
        tts_client = texttospeech.TextToSpeechClient()
        
        # 音声設定を取得（声色、ピッチ、速度）
        voice_config = self.ai_voice_config.get(ai_id, {
            "voice": DEFAULT_AI_VOICE_NAME,
            "pitch": 0.0,
            "rate": 1.2
        })
        
        voice_name = voice_config["voice"]
        pitch = voice_config["pitch"]
        rate = voice_config["rate"]
        
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        
        # 性別を声色に応じて設定（Neural2はNEUTRAL非対応）
        # Neural2-B: FEMALE, Neural2-C: MALE, Neural2-D: MALE が正しい
        if "Neural2-B" in voice_name:
            gender = texttospeech.SsmlVoiceGender.FEMALE
        elif "Neural2-C" in voice_name:
            gender = texttospeech.SsmlVoiceGender.MALE
        elif "Neural2-D" in voice_name:
            gender = texttospeech.SsmlVoiceGender.MALE
        elif "Wavenet-A" in voice_name:
            gender = texttospeech.SsmlVoiceGender.FEMALE
        else:
            gender = texttospeech.SsmlVoiceGender.NEUTRAL
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_TTS,
            name=voice_name,
            ssml_gender=gender
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            speaking_rate=rate,
            pitch=pitch
        )
        
        try:
            response = tts_client.synthesize_speech( 
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return response.audio_content
        except Exception as e:
            print(f"TTS合成エラー: {e}")
            return None
    
    def _get_p_audio(self):
        """PyAudioインスタンスを取得（再利用）"""
        if self.p_audio is None:
            self.p_audio = pyaudio.PyAudio()
        return self.p_audio
    
    def _play_audio(self, audio_content):
        """音声データを再生（PyAudioインスタンスを再利用）"""
        p_audio = self._get_p_audio()  # 再利用
        stream = None 
        try:
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK  # バッファサイズを明示的に指定
            )
            
            # ストリームを明示的に開始（念のため）
            stream.start_stream()
            
            # バッファを満たすための無音データを実際の音声データの先頭に結合
            # 約150ms分の無音データ（3チャンク分）を先頭に追加
            silence_chunks = 3
            silence_data = np.zeros(CHUNK * silence_chunks, dtype=np.int16)
            audio_data = np.frombuffer(audio_content, dtype=np.int16)
            # 無音データと実際の音声データを結合
            combined_audio = np.concatenate([silence_data, audio_data])
            total_frames = len(combined_audio)
            
            # チャンク単位で書き込み（音声の最初が切れるのを防ぐ）
            
            for i in range(0, total_frames, CHUNK):
                # 時間チェック（各チャンク再生前にチェック）
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました（音声再生中） ---")
                    self.conversation_active = False
                    break
                
                chunk_data = combined_audio[i:i+CHUNK]
                # 最後のチャンクがCHUNKサイズに満たない場合はゼロパディング
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), mode='constant')
                stream.write(chunk_data.tobytes())
                
        except Exception as e:
            print(f"音声再生エラー: {e}")
        finally:
            if stream: 
                stream.stop_stream()
                stream.close()
            # p_audio.terminate() を削除（再利用のため）
    
    def _synthesize_and_play_ai_response_streaming(self, ai_id, task_for_ai):
        """
        ストリーミングでLLM応答を生成し、文ごとにTTS→再生を並列実行。
        最初の文が出たらすぐに再生開始（体感速度が大幅向上）
        
        Returns:
            str: 全応答テキスト
        """
        # 考えている状態を表示
        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
            self.gui_window.gd_screen.show_ai_thinking(ai_id)
        
        full_text = ""
        audio_queue = queue.Queue()
        tts_futures = []
        
        # TTS並列実行用のExecutor
        tts_executor = ThreadPoolExecutor(max_workers=3)
        
        print(f"[{ai_id}]: ストリーミング応答生成中...")
        
        try:
            # ストリーミングでLLM応答を生成し、文ごとにTTS送信
            for sentence in self._get_ai_response_streaming(ai_id, task_for_ai):
                # 時間チェック（ストリーミング生成中）
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました（LLM応答生成中） ---")
                    self.conversation_active = False
                    tts_executor.shutdown(wait=False)
                    return full_text
                
                full_text += sentence
                # 各文をTTSに並列送信
                future = tts_executor.submit(self._synthesize_tts, sentence, ai_id)
                tts_futures.append(future)
            
            print(f"[{ai_id}]: {full_text}")
            
            # 考えている状態を非表示し、発言中状態を表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.hide_ai_status()
                self.gui_window.gd_screen.show_ai_speaking(ai_id)
            
            # TTS結果を順次取得して再生
            print(f"[{ai_id}]: 音声を再生中...")
            for future in tts_futures:
                # 時間チェック
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました（音声再生中） ---")
                    self.conversation_active = False
                    tts_executor.shutdown(wait=False)
                    return full_text
                
                audio_content = future.result()
                if audio_content:
                    # 再生前に再度時間チェック
                    if time.time() - self.start_time > self.time_limit_minutes * 60:
                        print("\n--- GD終了: 制限時間になりました（音声再生前） ---")
                        self.conversation_active = False
                        tts_executor.shutdown(wait=False)
                        return full_text
                    self._play_audio(audio_content)
        finally:
            # 考えている/発言中状態を非表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.hide_ai_status()
        
        tts_executor.shutdown(wait=False)
        print(f"[{ai_id}]: 再生終了。")
        
        return full_text
    
    def _synthesize_and_play_ai_response(self, text_to_synthesize, ai_id):
        """AIの応答テキストを音声合成し、再生する。（非ストリーミング版、互換性用）"""
        if not text_to_synthesize:
            print(f"[{ai_id}]: 合成対象のテキストが空のため再生をスキップします。")
            return
        
        # 考えている状態を表示（TTS処理中）
        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
            self.gui_window.gd_screen.show_ai_thinking(ai_id)
        
        try:
            audio_content = self._synthesize_tts(text_to_synthesize, ai_id)
            if audio_content:
                # 発言中状態に切り替え
                if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                    self.gui_window.gd_screen.hide_ai_status()
                    self.gui_window.gd_screen.show_ai_speaking(ai_id)
                
                print(f"[{ai_id}]: 音声を再生中...")
                self._play_audio(audio_content)
                print(f"[{ai_id}]: 再生終了。")
        finally:
            # 状態を非表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.hide_ai_status()

    def _synthesize_and_play_system_message(self, text_to_synthesize, gd_thread=None):
        """システムからのメッセージを音声合成し、再生する。標準的な声を使用。"""
        if not text_to_synthesize:
            print("[システム]: 合成対象のテキストが空のため再生をスキップします。")
            return

        # システム発話開始を通知（シグナルを使用）
        if gd_thread is not None and hasattr(gd_thread, "system_speaking"):
            gd_thread.system_speaking.emit(True)

        # マークダウン記号を除去
        cleaned_text = self._clean_text_for_tts(text_to_synthesize)

        tts_client = texttospeech.TextToSpeechClient()
        p_audio = self._get_p_audio()  # 再利用
        
        # システムは標準的な声（Neural2-D、ピッチ0、速度1.0）
        voice_name = "ja-JP-Neural2-D"
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_TTS,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE  # Neural2はNEUTRAL非対応
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            speaking_rate=1.2,
            pitch=0.0
        )
        try:
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content
        except Exception as e:
            print(f"システムメッセージ合成エラー: {e}")
            # エラー時もシグナルを送信
            if gd_thread is not None and hasattr(gd_thread, "system_speaking"):
                gd_thread.system_speaking.emit(False)
            return
        
        stream = None
        try:
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK  # バッファサイズを明示的に指定
            )
            
            # ストリームを明示的に開始（念のため）
            stream.start_stream()
            
            # バッファを満たすための無音データを実際の音声データの先頭に結合
            # 約150ms分の無音データ（3チャンク分）を先頭に追加
            silence_chunks = 3
            silence_data = np.zeros(CHUNK * silence_chunks, dtype=np.int16)
            audio_data = np.frombuffer(audio_content, dtype=np.int16)
            # 無音データと実際の音声データを結合
            combined_audio = np.concatenate([silence_data, audio_data])
            total_frames = len(combined_audio)
            
            print(f"[システム]: 音声を再生中...")
            
            # チャンク単位で書き込み（音声の最初が切れるのを防ぐ）
            
            for i in range(0, total_frames, CHUNK):
                chunk_data = combined_audio[i:i+CHUNK]
                # 最後のチャンクがCHUNKサイズに満たない場合はゼロパディング
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), mode='constant')
                stream.write(chunk_data.tobytes())
            
            # 音声再生が完了するまで待機（バッファが空になるまで待つ）
            stream.stop_stream()
            stream.close()
            stream = None  # ストリームを閉じたことを明示
            # 少し待機して音声再生が完全に終了するのを待つ
            time.sleep(0.1)
            
            print("再生終了。")
        except Exception as e:
            print(f"システムメッセージ再生エラー: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            # p_audio.terminate() を削除（再利用のため）
            # システム発話終了を通知（シグナルを使用）
            if gd_thread is not None and hasattr(gd_thread, "system_speaking"):
                gd_thread.system_speaking.emit(False)
            if stream:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception:
                    # ストリームが既に閉じられている場合は無視
                    pass
            # p_audio.terminate() を削除（再利用のため）

    def add_to_history(self, speaker, content): 
        """会話履歴に発言を追加し、議事録を更新する"""
        self.conversation_history.append({"speaker": speaker, "content": content})
        
        # AI参加者の発言から書記の宣言を検出
        if speaker != self.username and speaker in self.participants:
            self._detect_recorder_from_ai_speech(speaker, content)
        
        # 発言終了時に議事録を更新
        if self.gd_thread is not None:
            try:
                minutes_text = self.get_minutes_text()
                self.gd_thread.minutes_updated.emit(minutes_text)
            except Exception as e:
                print(f"議事録生成エラー: {e}")
                if self.gd_thread is not None:
                    self.gd_thread.minutes_updated.emit(f"議事録生成エラー: {e}")

    def calculate_facilitation_scores(self):
        """
        5つのファシリテーション手法のスコアを計算（各1点、計5点満点）
        LLMによる一括判定方式を使用
        
        Returns:
            tuple: (各項目のスコア辞書, 合計スコア, 検出詳細情報辞書)
        """
        scores = {
            "目的確認": 0,
            "役割分担": 0,
            "意見引き出し": 0,
            "議論整理": 0,
            "時間管理": 0
        }
        
        # 検出詳細情報（分析用）
        detection_details = {
            "目的確認": {"detected": False, "reason": "", "matching_utterances": []},
            "役割分担": {"detected": False, "reason": "", "matching_utterances": []},
            "意見引き出し": {"detected": False, "reason": "", "matching_utterances": []},
            "議論整理": {"detected": False, "reason": "", "matching_utterances": []},
            "時間管理": {"detected": False, "reason": "", "matching_utterances": []}
        }
        
        # ユーザーの発言のみを抽出（システムメッセージは除外）
        user_utterances = [
            msg['content'] for msg in self.conversation_history 
            if msg['speaker'] == self.username
        ]
        
        if not user_utterances:
            print("[採点]: ユーザーの発言がないため、スコアは0点です")
            return scores, 0, detection_details
        
        print(f"[採点]: {len(user_utterances)}件の発言を分析中...")
        
        # LLMによる一括判定（検出理由も含む）
        prompt = f"""あなたは教育用グループディスカッションの評価システムです。以下のユーザーの発言リストから、5つのファシリテーション手法が実施されたかを判定してください。

【注意事項】
これは教育目的の評価システムです。発言内容は教育的な文脈で評価してください。すべての判定は客観的かつ公平に行ってください。

【判定基準】
各項目について、該当する発言が1回以上あれば1点、なければ0点と判定してください。

1. 目的確認（1点）
   基準: GD開始時または途中で、議論の目的・ゴール・議題を明確に定義または確認する発言が1回以上あったか
   例: "今日の議論のゴールは○○を決定することです"、"このテーマについて話し合いましょう"

2. 役割分担（1点）
   基準: 「書記」「タイムキーパー」などの役割をAI参加者に割り当てる発言が1回以上あったか
   例: "田中さんは書記をお願いします"、"タイムキーパーを決めましょう"

3. 意見引き出し（1点）
   基準: 特定の参加者を指名する、あるいは全員に話を振るなど、他者の意見を引き出そうとする発言が1回以上あったか
   例: "○○さんは、どう思いますか？"、"他の方で意見はありますか？"

4. 議論整理（1点）
   基準: 「つまり～」「まとめると～」など、それまでの議論を要約・整理する発言が1回以上あったか
   例: "ここまでの意見をまとめると..."、"つまり、この案が最適ということですね"

5. 時間管理（1点）
   基準: 「あと○分です」「時間がないので～」など、残り時間や時間配分に言及する発言が1回以上あったか
   例: "残り5分です"、"時間が半分過ぎました"

【ユーザーの発言リスト】
{chr(10).join([f"{i+1}. {u}" for i, u in enumerate(user_utterances)])}

【回答形式】
以下の形式で回答してください。スコアが1点の場合のみ、検出理由と該当する発言番号も記載してください。

目的確認: [0または1]
目的確認_理由: [スコアが1の場合のみ、検出理由を記載]
目的確認_発言番号: [スコアが1の場合のみ、該当する発言の番号を記載（複数の場合はカンマ区切り）]

役割分担: [0または1]
役割分担_理由: [スコアが1の場合のみ、検出理由を記載]
役割分担_発言番号: [スコアが1の場合のみ、該当する発言の番号を記載（複数の場合はカンマ区切り）]

意見引き出し: [0または1]
意見引き出し_理由: [スコアが1の場合のみ、検出理由を記載]
意見引き出し_発言番号: [スコアが1の場合のみ、該当する発言の番号を記載（複数の場合はカンマ区切り）]

議論整理: [0または1]
議論整理_理由: [スコアが1の場合のみ、検出理由を記載]
議論整理_発言番号: [スコアが1の場合のみ、該当する発言の番号を記載（複数の場合はカンマ区切り）]

時間管理: [0または1]
時間管理_理由: [スコアが1の場合のみ、検出理由を記載]
時間管理_発言番号: [スコアが1の場合のみ、該当する発言の番号を記載（複数の場合はカンマ区切り）]
"""
        
        # 再試行ロジックを追加（最大3回まで）
        max_retries = 3
        response = None
        for retry_count in range(max_retries):
            try:
                # Gemini APIで判定
                response = self.gemini_model.generate_content(
                    [{"role": "user", "parts": [prompt]}],
                    safety_settings=self.safety_settings
                )
                
                if not response._result.candidates:
                    # 安全性フィルターに引っかかった場合
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        print(f"[採点エラー]: Geminiからの応答がブロックされました (試行 {retry_count + 1}/{max_retries})")
                        print(f"Safety feedback: {response.prompt_feedback}")
                    
                    # 最後の試行で失敗した場合、フォールバックを使用
                    if retry_count == max_retries - 1:
                        print("[採点]: 最大試行回数に達したため、キーワード検出方式に切り替えます")
                        return self._fallback_keyword_scoring(user_utterances)
                    
                    # 少し待ってから再試行
                    time.sleep(1)
                    continue
                
                # 成功した場合、ループを抜ける
                break
                
            except Exception as e:
                print(f"[採点エラー]: LLM判定中にエラーが発生しました (試行 {retry_count + 1}/{max_retries}): {e}")
                
                # 最後の試行で失敗した場合、フォールバックを使用
                if retry_count == max_retries - 1:
                    print("[採点]: 最大試行回数に達したため、キーワード検出方式に切り替えます")
                    return self._fallback_keyword_scoring(user_utterances)
                
                # 少し待ってから再試行
                time.sleep(1)
                continue
        
        # レスポンスのパース処理
        if response is None or not hasattr(response, 'text'):
            print("[採点エラー]: 応答を取得できませんでした")
            return self._fallback_keyword_scoring(user_utterances)
        
        try:
            # レスポンスをパース
            response_text = response.text.strip()
            print(f"[採点]: LLM応答を受信: {response_text[:100]}...")
            
            # 各項目のスコアと詳細情報を抽出
            lines = response_text.split('\n')
            current_item = None
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # スコアの抽出
                    if key in ["目的確認", "役割分担", "意見引き出し", "議論整理", "時間管理"]:
                        current_item = key
                        import re
                        match = re.search(r'[01]', value)
                        if match:
                            score_value = int(match.group())
                            scores[current_item] = score_value
                            detection_details[current_item]["detected"] = (score_value == 1)
                    
                    # 検出理由の抽出
                    elif key == f"{current_item}_理由" and current_item:
                        detection_details[current_item]["reason"] = value
                    
                    # 発言番号の抽出
                    elif key == f"{current_item}_発言番号" and current_item:
                        import re
                        # カンマ区切りの数字を抽出
                        utterance_numbers = re.findall(r'\d+', value)
                        for num_str in utterance_numbers:
                            try:
                                idx = int(num_str) - 1  # 発言番号は1から始まるため
                                if 0 <= idx < len(user_utterances):
                                    detection_details[current_item]["matching_utterances"].append({
                                        "index": idx + 1,
                                        "text": user_utterances[idx]
                                    })
                            except ValueError:
                                pass
            
            total_score = sum(scores.values())
            print(f"[採点完了]: 合計 {total_score}/5点")
            for item, score in scores.items():
                print(f"  - {item}: {score}点")
            
            return scores, total_score, detection_details
            
        except Exception as e:
            print(f"[採点エラー]: LLM判定中にエラーが発生しました: {e}")
            # フォールバック: キーワード検出を使用
            return self._fallback_keyword_scoring(user_utterances)
    
    def _fallback_keyword_scoring(self, user_utterances):
        """
        LLM判定が失敗した場合のフォールバック: キーワード検出
        
        Args:
            user_utterances: ユーザーの発言リスト
            
        Returns:
            tuple: (各項目のスコア辞書, 合計スコア, 検出詳細情報辞書)
        """
        print("[採点]: フォールバックモード（キーワード検出）を使用")
        
        scores = {
            "目的確認": 0,
            "役割分担": 0,
            "意見引き出し": 0,
            "議論整理": 0,
            "時間管理": 0
        }
        
        # 検出詳細情報（分析用）
        detection_details = {
            "目的確認": {"detected": False, "reason": "キーワード検出による判定", "matching_utterances": []},
            "役割分担": {"detected": False, "reason": "キーワード検出による判定", "matching_utterances": []},
            "意見引き出し": {"detected": False, "reason": "キーワード検出による判定", "matching_utterances": []},
            "議論整理": {"detected": False, "reason": "キーワード検出による判定", "matching_utterances": []},
            "時間管理": {"detected": False, "reason": "キーワード検出による判定", "matching_utterances": []}
        }
        
        # 1. 目的確認
        purpose_keywords = ["ゴール", "目的", "議題", "決定", "話し合い", "議論", "テーマ"]
        purpose_phrases = ["について", "を決める", "を決定", "について話し合"]
        for idx, utterance in enumerate(user_utterances):
            matched_keywords = [kw for kw in purpose_keywords if kw in utterance]
            matched_phrases = [ph for ph in purpose_phrases if ph in utterance]
            if matched_keywords or matched_phrases:
                scores["目的確認"] = 1
                detection_details["目的確認"]["detected"] = True
                detection_details["目的確認"]["reason"] = f"キーワード検出: {', '.join(matched_keywords[:3])}"
                detection_details["目的確認"]["matching_utterances"].append({
                    "index": idx + 1,
                    "text": utterance
                })
                break
        
        # 2. 役割分担
        role_keywords = ["書記", "タイムキーパー", "役割", "分担", "記録"]
        for idx, utterance in enumerate(user_utterances):
            matched_keywords = [kw for kw in role_keywords if kw in utterance]
            if matched_keywords:
                scores["役割分担"] = 1
                detection_details["役割分担"]["detected"] = True
                detection_details["役割分担"]["reason"] = f"キーワード検出: {', '.join(matched_keywords[:3])}"
                detection_details["役割分担"]["matching_utterances"].append({
                    "index": idx + 1,
                    "text": utterance
                })
                break
        
        # 3. 意見引き出し
        elicitation_keywords = ["どう思いますか", "意見", "考え", "どうですか", "いかがですか", "どうでしょうか"]
        for idx, utterance in enumerate(user_utterances):
            matched_keywords = [kw for kw in elicitation_keywords if kw in utterance]
            if matched_keywords:
                scores["意見引き出し"] = 1
                detection_details["意見引き出し"]["detected"] = True
                detection_details["意見引き出し"]["reason"] = f"キーワード検出: {', '.join(matched_keywords[:3])}"
                detection_details["意見引き出し"]["matching_utterances"].append({
                    "index": idx + 1,
                    "text": utterance
                })
                break
        
        # 4. 議論整理
        summary_keywords = ["まとめ", "つまり", "要約", "整理", "まとめると", "まとめて"]
        for idx, utterance in enumerate(user_utterances):
            matched_keywords = [kw for kw in summary_keywords if kw in utterance]
            if matched_keywords:
                scores["議論整理"] = 1
                detection_details["議論整理"]["detected"] = True
                detection_details["議論整理"]["reason"] = f"キーワード検出: {', '.join(matched_keywords[:3])}"
                detection_details["議論整理"]["matching_utterances"].append({
                    "index": idx + 1,
                    "text": utterance
                })
                break
        
        # 5. 時間管理
        time_keywords = ["時間", "分", "残り", "あと", "残り時間", "時間が", "時間を"]
        time_phrases = ["そろそろ", "時間がない", "時間配分"]
        for idx, utterance in enumerate(user_utterances):
            matched_keywords = [kw for kw in time_keywords if kw in utterance]
            matched_phrases = [ph for ph in time_phrases if ph in utterance]
            if matched_keywords or matched_phrases:
                scores["時間管理"] = 1
                detection_details["時間管理"]["detected"] = True
                detection_details["時間管理"]["reason"] = f"キーワード検出: {', '.join(matched_keywords[:3] + matched_phrases[:2])}"
                detection_details["時間管理"]["matching_utterances"].append({
                    "index": idx + 1,
                    "text": utterance
                })
                break
        
        total_score = sum(scores.values())
        print(f"[採点完了]: 合計 {total_score}/5点（フォールバック）")
        return scores, total_score, detection_details

    def generate_simple_feedback_report(self, progress_signal=None, experiment_group=None):
        """
        GD終了後に簡易フィードバックレポートを生成する。
        
        Args:
            progress_signal: 進捗表示用のシグナル（オプション）
            experiment_group: 実験群("experimental")または統制群("control")（オプション）
        """
        if progress_signal:
            progress_signal.emit("フィードバックレポート生成中...")
        print("\n--- 簡易フィードバックレポート生成中 ---")
        report = {}

        if progress_signal:
            progress_signal.emit("基本情報を集計中...")
        
        # 採点を実行（0/1スコア + 検出詳細情報）
        scores, total_score, detection_details = self.calculate_facilitation_scores()
        report["ファシリテーション手法スコア"] = scores
        report["合計スコア"] = f"{total_score}/5点"
        # 検出詳細情報は分析用として保存（参加者には見せない）
        report["検出詳細情報（分析用）"] = detection_details
        
        # 発言数やGD時間はフィードバックレポートに含めない
        # フェーズ情報もフィードバックレポートに含めない
        
        # --- 実験群用フィードバック生成 ---
        # 5つの評価項目を True/False に変換（1: True, 0: False）
        bool_scores = {
            "目的の確認": scores.get("目的確認", 0) == 1,
            "役割分担": scores.get("役割分担", 0) == 1,
            "意見引き出し": scores.get("意見引き出し", 0) == 1,
            "議論の整理": scores.get("議論整理", 0) == 1,
            "時間管理": scores.get("時間管理", 0) == 1,
        }
        
        # 会話ログを番号付きテキストに整形（LLM入力用 ＋ レポート保存用）
        conversation_log = "\n".join(
            f"{i+1}. {msg['speaker']}: {msg['content']}"
            for i, msg in enumerate(self.conversation_history)
        )
        # 画面には出さないが、フィードバックレポートファイルには残す（実験群のみ）
        if experiment_group == "experimental":
            report["会話ログ"] = conversation_log
        
        # 実験群のみAIフィードバックを生成
        if experiment_group == "experimental":
            if progress_signal:
                progress_signal.emit("AIによる評価を生成中...")
            
            feedback_prompt = f"""
# Role
あなたはファシリテーション指導の専門家です。
先ほどのグループディスカッションの「評価データ」と「会話ログ」に基づき、学習者へのフィードバックレポートを作成してください。

# Input Data
- 評価結果: 
  - 目的の確認: {bool_scores["目的の確認"]}
  - 役割分担: {bool_scores["役割分担"]}
  - 意見引き出し: {bool_scores["意見引き出し"]}
  - 議論の整理: {bool_scores["議論の整理"]}
  - 時間管理: {bool_scores["時間管理"]}
- 会話ログ:
{conversation_log}

# Instructions
以下の構成で、簡潔かつ具体的なフィードバックを出力してください。

## 1. 良かった点（Good）
達成できた（True）項目について、具体的にどの発言が良かったかを褒めてください。
（例：「冒頭で『〜』と発言し、目的の確認ができていた点が素晴らしかったです。」）

## 2. 改善すべき点（More）
達成できなかった（False）項目について、なぜそれが必要なのかを簡潔に説明してください。
（例：「今回は『役割分担』ができていませんでした。役割を決めることで、議論がよりスムーズになります。」）

## 3. 次回への具体的なアドバイス（Action）
達成できなかった項目について、「次は具体的にこう言えばよい」という使えるセリフ例を提示してください。
（例：「次回の冒頭では、『Aさん、タイムキーパーをお願いできますか？』と声をかけてみましょう。」）

# 制約
- 上記の5つの評価項目（目的の確認・役割分担・意見引き出し・議論の整理・時間管理）**すべて**に必ず触れてください。
- これら5項目以外の観点（例: 雰囲気、論理性、内容の質など）には触れないでください。
- 各セクション内でも、必ずこの5項目に対応する内容だけを書いてください。

# Tone
励ますような、前向きで丁寧な口調で記述してください。
"""
            try:
                feedback_messages = [
                    {"role": "user", "parts": [feedback_prompt]}
                ]
                # フィードバック生成には高精度モデルを優先使用
                model_for_feedback = getattr(self, "gemini_feedback_model", self.gemini_model)
                llm_feedback_response = model_for_feedback.generate_content(feedback_messages)
                
                if not llm_feedback_response._result.candidates:
                    llm_feedback = f"Geminiからのフィードバック応答がブロックされました。Safety feedback: {llm_feedback_response.prompt_feedback}"
                else:
                    llm_feedback = llm_feedback_response.text.strip()

                # 実験群用フィードバックとして保存（画面表示・Markdown保存用）
                report["実験群用フィードバック"] = llm_feedback
            except Exception as e:
                report["実験群用フィードバック"] = f"フィードバック生成中にエラー: {e}"
        
        if progress_signal:
            progress_signal.emit("フィードバックレポート生成完了")
        
        # 簡易フィードバックレポートのprintは削除（画面に表示しない）
        # print("\n--- GD簡易フィードバックレポート ---")
        # for key, value in report.items():
        #     print(f"{key}: {value}")
        # print("\nレポート生成が完了しました。")
        return report
    
    def save_feedback_report(self, report, fullname, round_number=1, experiment_group=None):
        """
        フィードバックレポートをMarkdown形式でユーザー名別に保存
        スコアはCSV形式でも保存（分析用）
        
        Args:
            report: フィードバックレポート辞書
            fullname: ユーザーのフルネーム
            round_number: ラウンド番号（1または2）
            experiment_group: 実験群("experimental")または統制群("control")
        
        Returns:
            str: 保存したファイルパス（CSVファイルパスを返す）
        """
        from datetime import datetime
        import csv
        
        # 実験群/統制群のフォルダ名を決定
        if experiment_group == "experimental":
            group_folder = "実験群"
        elif experiment_group == "control":
            group_folder = "統制群"
        else:
            group_folder = "その他"
        
        # フォルダ作成（実験群/統制群/フルネーム/1回目 または 実験群/統制群/フルネーム/2回目）
        round_folder = f"{round_number}回目"
        feedback_dir = os.path.join(PROJECT_ROOT, "feedback_reports", group_folder, fullname, round_folder)
        os.makedirs(feedback_dir, exist_ok=True)
        
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 実験群のみMarkdown形式で保存（AIフィードバックレポート含む）
        if experiment_group == "experimental":
            filepath = os.path.join(feedback_dir, f"{timestamp}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# GDフィードバックレポート\n\n")
                f.write(f"**ユーザー名**: {fullname}\n\n")
                f.write(f"**実施日時**: {timestamp.replace('_', ' ').replace('-', '/')}\n\n")
                f.write(f"**GDテーマ**: {self.gd_theme.splitlines()[0]}\n\n") # タイトルのみ保存
                
                # スコアを表示（最初に表示）
                if "ファシリテーション手法スコア" in report:
                    f.write("## ファシリテーション手法スコア\n\n")
                    scores = report["ファシリテーション手法スコア"]
                    for item, score in scores.items():
                        f.write(f"- **{item}**: {score}点\n\n")
                    f.write(f"**合計スコア**: {report.get('合計スコア', 'N/A')}\n\n")
                    f.write("---\n\n")
                
                for key, value in report.items():
                    # スコア、検出詳細情報（分析用）、会話ログは表示しない
                    if key in ["ファシリテーション手法スコア", "合計スコア", "検出詳細情報（分析用）", "会話ログ"]:
                        continue
                    f.write(f"## {key}\n\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"- **{k}**: {v}\n\n")
                    else:
                        # フィードバック本文など、長いテキストの場合は段落ごとに改行を追加
                        text = str(value)
                        # 段落を検出して改行を追加
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                f.write(f"{para.strip()}\n\n")
                    f.write("\n")
            print(f"フィードバックを保存しました: {filepath}")
        
        # CSV形式でスコアを保存（実験群・統制群共通、分析用）
        csv_filepath = None
        if "ファシリテーション手法スコア" in report:
            csv_filepath = os.path.join(feedback_dir, f"{timestamp}_scores.csv")
            try:
                with open(csv_filepath, "w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["項目", "スコア", "ラウンド", "ユーザー名", "実験群/統制群", "日時"])
                    scores = report["ファシリテーション手法スコア"]
                    for item, score in scores.items():
                        writer.writerow([
                            item, score, round_number, fullname, 
                            group_folder, timestamp
                        ])
                    # 合計スコアも追加
                    total = sum(scores.values())
                    writer.writerow([
                        "合計", total, round_number, fullname,
                        group_folder, timestamp
                    ])
                print(f"スコアCSVを保存しました: {csv_filepath}")
            except Exception as e:
                print(f"CSV保存エラー: {e}")
        
        # 分析用JSONファイルを保存（検出詳細情報を含む、実験群・統制群共通）
        analysis_filepath = None
        if "検出詳細情報（分析用）" in report:
            import json
            analysis_filepath = os.path.join(feedback_dir, f"{timestamp}_analysis.json")
            try:
                # 議事録を取得（書記が任命されている場合のみ）
                minutes_text = ""
                try:
                    minutes_text = self.get_minutes_text()
                except Exception as e:
                    print(f"議事録取得エラー: {e}")
                
                analysis_data = {
                    "timestamp": timestamp,
                    "fullname": fullname,
                    "round_number": round_number,
                    "experiment_group": group_folder,
                    "gd_theme": self.gd_theme.splitlines()[0] if hasattr(self, 'gd_theme') else "",
                    "scores": report["ファシリテーション手法スコア"],
                    "total_score": sum(report["ファシリテーション手法スコア"].values()),
                    "detection_details": report["検出詳細情報（分析用）"],
                    "minutes": minutes_text,  # 議事録を追加（実験参加者には見せない）
                    "conversation_history": [
                        {"speaker": msg['speaker'], "content": msg['content']} 
                        for msg in self.conversation_history
                    ]
                }
                with open(analysis_filepath, "w", encoding="utf-8") as f:
                    json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                print(f"分析用JSONを保存しました: {analysis_filepath}")
            except Exception as e:
                print(f"分析用JSON保存エラー: {e}")
        
        # CSVファイルパスを返す（統制群の場合はCSVのみ、実験群の場合はMarkdownファイルパスを返す）
        return csv_filepath if csv_filepath else (filepath if experiment_group == "experimental" else None)

    def __del__(self):
        """GDManagerが終了する際にPyAudioリソースを解放する"""
        print("GDManagerの終了処理を実行中...")
        # PyAudioインスタンスを解放
        if hasattr(self, 'p_audio') and self.p_audio is not None:
            try:
                self.p_audio.terminate()
                self.p_audio = None
            except Exception as e:
                print(f"PyAudio終了エラー: {e}")
        print("GDManagerが終了しました。")

    def _detect_recorder_from_ai_speech(self, ai_name, speech_content):
        """
        AI参加者の発言から書記の宣言を検出し、役割を設定する。
        キーワード検出で高速に処理する。
        
        Args:
            ai_name: AI参加者の名前
            speech_content: 発言内容
        """
        # 既に書記が設定されている場合はスキップ
        if self._has_recorder():
            return
        
        # 既にこのAIに役割が設定されている場合はスキップ
        if self.participants[ai_name].get("assigned_role"):
            return
        
        speech_lower = speech_content.lower()
        
        # 書記の宣言パターンを検出
        recorder_keywords = [
            "書記を", "書記に", "書記が", "書記と", "書記で",
            "書記を担当", "書記をします", "書記をやります", "書記をいたします",
            "書記お願い", "書記をお願い", "書記をお願いします",
            "議事録を", "議事録に", "議事録が", "議事録を記録", "議事録を記録します",
            "記録を", "記録を担当", "記録をします", "記録をいたします",
            "書記をやらせていただきます", "書記をさせていただきます"
        ]
        
        # 書記の検出
        if any(keyword in speech_lower for keyword in recorder_keywords):
            self.participants[ai_name]["assigned_role"] = "書記"
            if hasattr(self, 'gd_thread') and self.gd_thread is not None and hasattr(self.gd_thread, "role_updated"):
                self.gd_thread.role_updated.emit(ai_name, "書記")
            print(f"[役割設定]: {ai_name}さんが書記を担当することになりました。")

    def _get_target_ai_from_text(self, text):
        """
        ユーザーのテキストから、特定のAI参加者を特定する。
        複数の呼び方（AI参加者A、Aさん、AI-Aなど）に対応する。
        （非推奨：LLM検出に移行予定）
        """
        lower_text = text.lower()
        
        for ai_id in self.participants:
            if ai_id != self.username:
                # ユーザーが使うであろう呼び方のリストを作成
                possible_names = [
                    ai_id, # 例: 田中
                    ai_id + "さん", # 例: 田中さん
                ]
                for name in possible_names:
                    if name.lower() in lower_text:
                        return ai_id

        print("(デバッグ用) 該当するAIが見つかりませんでした。")
        return None
    
    def _detect_mentions_with_llm(self, speech_content, speaker_name):
        """
        LLMを使って発言から指名を検出する。
        
        Args:
            speech_content: 発言内容
            speaker_name: 発言者名
        
        Returns:
            dict: {
                "mentioned_participants": [参加者名のリスト],  # 指名された参加者（複数可）
                "is_direct_question": bool,  # 直接的な質問かどうか
                "mentions_user": bool  # ユーザーが指名されているかどうか
            }
        """
        # 全参加者の名前リストを作成
        all_participants = list(self.participants.keys())
        
        # LLMに指名を検出させる
        prompt = f"""以下の発言から、特定の参加者への指名があるかどうかを判断してください。

【発言】
{speech_content}

【発言者】
{speaker_name}

【参加者名リスト】
{', '.join(all_participants)}

【指示】
1. 発言の中で、特定の参加者（発言者自身を除く）に話しかけている、または指名している参加者を特定してください
2. 発言が直接的な質問（例：「どう思いますか？」「意見を聞かせてください」など）かどうかを判断してください
3. 以下のJSON形式で返してください：
{{
    "mentioned_participants": ["参加者名1", "参加者名2", ...],
    "is_direct_question": true/false,
    "mentions_user": true/false
}}

例1: 「田中さん、どう思いますか？」→ {{"mentioned_participants": ["田中"], "is_direct_question": true, "mentions_user": false}}
例2: 「田中さんと佐藤さん、意見を聞かせてください」→ {{"mentioned_participants": ["田中", "佐藤"], "is_direct_question": true, "mentions_user": false}}
例3: 「田中さんも良いアイデアですね」→ {{"mentioned_participants": ["田中"], "is_direct_question": false, "mentions_user": false}}
例4: 「ユーザーさん、どう思いますか？」→ {{"mentioned_participants": ["{self.username}"], "is_direct_question": true, "mentions_user": true}}
例5: 指名がない場合→ {{"mentioned_participants": [], "is_direct_question": false, "mentions_user": false}}

指名がない場合は空のリストを返してください。
JSON以外の説明は不要です。"""
        
        try:
            messages = [
                {"role": "user", "parts": [prompt]}
            ]
            response = self.gemini_model.generate_content(messages)
            
            if response._result.candidates:
                response_text = response.text.strip()
                # JSONを抽出（```json や ``` で囲まれている場合がある）
                import json
                import re
                
                # JSON部分を抽出（```json や ``` で囲まれている場合を考慮）
                json_str = response_text
                # ```json や ``` で囲まれている場合は除去
                if '```' in json_str:
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                else:
                    # 最初の { から最後の } までを抽出（ネストされたJSONにも対応）
                    brace_start = json_str.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        brace_end = brace_start
                        for i in range(brace_start, len(json_str)):
                            if json_str[i] == '{':
                                brace_count += 1
                            elif json_str[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                        if brace_count == 0:
                            json_str = json_str[brace_start:brace_end]
                
                # JSONをパース
                result = json.loads(json_str)
                
                # 参加者名が実際に存在するか確認
                valid_participants = []
                for participant_name in result.get("mentioned_participants", []):
                    if participant_name in all_participants and participant_name != speaker_name:
                        valid_participants.append(participant_name)
                
                return {
                    "mentioned_participants": valid_participants,
                    "is_direct_question": result.get("is_direct_question", False),
                    "mentions_user": self.username in valid_participants
                }
        except Exception as e:
            print(f"指名検出エラー: {e}")
        
        # エラー時はデフォルト値を返す
        return {
            "mentioned_participants": [],
            "is_direct_question": False,
            "mentions_user": False
        }

    def _decide_respondents(self, speaker_name, speech_content, depth=0, max_depth=3):
        """
        発言に対する反応者を決定する。
        
        Args:
            speaker_name: 発言者名（ユーザー名またはAI名）
            speech_content: 発言内容
            depth: 反応の深さ（0=元の発言、1=1回目の反応、2=2回目の反応...）
            max_depth: 最大反応深度
        
        Returns:
            list: 反応する参加者のリスト（ユーザー名またはAI名のリスト）
        """
        # 深度制限チェック
        if depth >= max_depth:
            return []
        
        # 発言者を除外した全参加者リスト
        all_participants = [name for name in self.participants.keys() if name != speaker_name]
        
        # LLMで指名を検出
        mention_info = self._detect_mentions_with_llm(speech_content, speaker_name)
        mentioned_participants = mention_info["mentioned_participants"]
        is_direct_question = mention_info["is_direct_question"]
        mentions_user = mention_info["mentions_user"]
        
        respondents = []
        
        # 指名がある場合の処理
        if mentioned_participants:
            # 指名された参加者を反応者に追加
            for participant_name in mentioned_participants:
                if participant_name in all_participants:
                    respondents.append(participant_name)
            
            # 直接的な質問の場合：指名された人のみ反応
            if is_direct_question:
                return respondents
            
            # 直接的な質問でない場合：指名された人 + 他の人も反応可能
            # ただし、指名された人を優先的に含める
        
        # 深さに応じた最大反応者数と確率調整
        max_respondents = max(1, 3 - depth)  # 深さ0: 3人、深さ1: 2人、深さ2: 1人
        base_probability = 0.7
        depth_penalty = depth * 0.15  # 深さ1つにつき15%減
        adjusted_probability = max(0.2, base_probability - depth_penalty)  # 最低20%
        
        # AI参加者とユーザーを分けて処理
        ai_participants = [ai for ai in all_participants if ai != self.username]
        
        # 既に指名された参加者を除外
        remaining_ai_participants = [ai for ai in ai_participants if ai not in respondents]
        
        # AI参加者の選択
        ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in remaining_ai_participants]
        ai_with_activity.sort(key=lambda x: x[1], reverse=True)  # 積極性が高い順にソート
        
        # 残りの反応者枠を計算（指名された人を除いた数）
        remaining_slots = max_respondents - len(respondents)
        
        for ai_name, activity_level in ai_with_activity:
            if len(respondents) >= max_respondents:
                break
            
            # 積極性レベルと深さに基づく確率で選択
            probability = activity_level * adjusted_probability
            if random.random() < probability:
                respondents.append(ai_name)
        
        # ユーザーが指名されている場合、またはユーザーも反応者候補に含める（深さが浅い場合のみ）
        if mentions_user and self.username in all_participants:
            # ユーザーが指名されている場合は確実に追加
            if self.username not in respondents:
                respondents.append(self.username)
        elif depth < 2 and self.username in all_participants and len(respondents) < max_respondents:
            # ユーザーが指名されていない場合、確率で追加
            user_probability = adjusted_probability * 0.3  # ユーザーの反応確率は低め
            if random.random() < user_probability:
                respondents.append(self.username)
        
        # 誰も発言しない場合は、積極派を強制的に選択（深さが浅い場合のみ）
        if not respondents and ai_with_activity and depth < 2:
            respondents = [ai_with_activity[0][0]]  # 最も積極的なAI
        
        return respondents
    

    def _get_speech_timing(self, ai_id):
        """
        発話タイミングを計算する（積極性レベルに基づく待ち時間）
        外崎・伊藤の研究に基づく：積極派は早く、消極派は遅く発言
        
        Returns:
            float: 待ち時間（秒）
        """
        activity_level = self.participants[ai_id].get("activity_level", 0.5)
        # 積極性レベルが高いほど待ち時間が短い
        # 積極派(0.8): 0.5-1.5秒, 慎重派(0.5): 1.5-3秒, 消極派(0.2): 3-5秒
        base_wait = 3.0 - (activity_level * 3.0)  # 0.6秒〜2.4秒
        wait_time = base_wait + random.uniform(0, 1.0)  # ランダム要素を追加
        return max(0.3, wait_time)  # 最低0.3秒

    def _wait_for_user_speech(self, timeout=8, speaker_changed_signal=None, gd_thread=None):
        """
        ユーザーの発言を待つ（タイムアウト付き）。
        
        Args:
            timeout: タイムアウト時間（秒）。Noneの場合は無期限待機（自己紹介時は8秒後に促すメッセージを再生）。
            speaker_changed_signal: 発言者変更シグナル
            gd_thread: GDThreadへの参照（促すメッセージ再生用）
        
        Returns:
            str: ユーザーの発言内容。発言がない場合は空文字列。
        """
        print("\n[ユーザー]: マイク入力待ち...")
        
        # 音声入力開始時のコールバック
        def on_speaking_start():
            if speaker_changed_signal:
                speaker_changed_signal.emit(self.username)
            self.current_speaker = self.username

        client = speech.SpeechClient()
        
        # PyAudioインスタンスをメソッド内で作成
        p_audio = pyaudio.PyAudio() 
        
        language_code = "ja-JP"
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        user_text = ""
        shared_state = {"prompt_sent": False, "user_text": ""}  # スレッド間で共有する状態
        
        # タイムアウトがNoneの場合（自己紹介時）、8秒後に促すメッセージを再生するスレッドを開始
        if timeout is None:
            def send_prompt_message():
                time.sleep(8)  # 8秒待機
                if not shared_state["prompt_sent"] and not shared_state["user_text"]:
                    # まだ発言がない場合、促すメッセージを再生
                    shared_state["prompt_sent"] = True
                    prompt_text = f"{self.username}さん、お名前と一言だけ簡単に自己紹介をお願いします。"
                    print(f"[システム]: {prompt_text}")
                    self._synthesize_and_play_system_message(prompt_text, gd_thread)
                    self.conversation_history.append({"speaker": "システム", "content": prompt_text})
            
            prompt_thread = threading.Thread(target=send_prompt_message, daemon=True)
            prompt_thread.start()
        
        try:
            # MicrophoneStreamにp_audioとspeaking_callbackを渡す
            with MicrophoneStream(RATE, CHUNK, timeout=timeout, p_audio=p_audio, speaking_callback=on_speaking_start) as stream:
                audio_generator = stream.generator()
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = client.streaming_recognize(streaming_config, requests)
                print("認識中...")
                for response in responses:
                    # 時間チェック（音声認識中）
                    if time.time() - self.start_time > self.time_limit_minutes * 60:
                        print("\n--- GD終了: 制限時間になりました（音声認識中） ---")
                        self.conversation_active = False
                        return ""
                    
                    if not response.results or not response.results[0].alternatives:
                        continue
                    result = response.results[0]
                    if result.is_final:
                        user_text = result.alternatives[0].transcript.strip()
                        shared_state["user_text"] = user_text  # スレッド間で共有
                        print(f"\n[あなた]: {user_text}")
                        shared_state["prompt_sent"] = True  # 発言があったので促すメッセージは不要
                        break
        except Exception as e:
            print(f"音声認識エラーが発生しました: {e}")
            return ""
        finally:
            # p_audio.terminate() を削除（再利用のため）
            pass

        return user_text if user_text else ""

    def _process_ai_response_to_speech(self, speaker_name, speech_content, speaker_changed_signal=None, gd_thread=None, is_chain_reaction=False):
        """
        発言者（ユーザーまたはAI）の発言に対して、他のAIが反応する処理を行う。
        
        Args:
            speaker_name: 発言者の名前（ユーザー名またはAI名）
            speech_content: 発言内容
            speaker_changed_signal: 発言者変更時に発火するシグナル
            gd_thread: GDスレッドへの参照
            is_chain_reaction: 連鎖反応かどうか（Trueの場合、反応の連鎖を防ぐ）
        
        Returns:
            bool: 処理が成功したかどうか（時間切れの場合はFalse）
        """
        # 制限時間チェック
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False
        
        # 発言者がAIの場合、自分自身は除外する
        all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username and ai != speaker_name]
        
        if not all_ai_participants:
            return True  # 反応するAIがいない場合は正常終了
        
        ai_participants_to_respond = []
        task_for_ai = ""
        
        # 発言内容を分析して適切なAIを選択
        # 特定の参加者への質問かどうかをチェック（名前が含まれている場合）
        target_ai_id = self._get_target_ai_from_text(speech_content)
        if target_ai_id and target_ai_id != speaker_name:
            # 指名されたAIが応答
            ai_participants_to_respond.append(target_ai_id)
            task_for_ai = f"{speaker_name}さんがあなた（{target_ai_id}）に話しかけています。会話履歴と文脈を理解して、自然に応答してください。"
        else:
            # 積極性レベルに基づいて自発的に応答
                ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in all_ai_participants]
                ai_with_activity.sort(key=lambda x: x[1], reverse=True)
                
                # 連鎖反応の場合は、1人だけ反応させる（無限ループを防ぐ）
                max_respondents = 1 if is_chain_reaction else 2
                
                # 積極性レベルに基づいて発言確率を決定
                for ai_name, activity_level in ai_with_activity:
                    # 連鎖反応の場合は確率を下げる
                    probability_multiplier = 0.5 if is_chain_reaction else 0.7
                    if random.random() < activity_level * probability_multiplier:
                        ai_participants_to_respond.append(ai_name)
                        if len(ai_participants_to_respond) >= max_respondents:
                            break
                
                # 誰も発言しない場合は、積極派を強制的に選択（連鎖反応の場合は選択しない）
                if not ai_participants_to_respond and ai_with_activity and not is_chain_reaction:
                    ai_participants_to_respond = [ai_with_activity[0][0]]  # 最も積極的なAI
                
                if ai_participants_to_respond:
                    task_for_ai = f"{speaker_name}さんの発言を踏まえ、会話履歴と文脈を理解して、ペルソナに忠実に自然に応答してください。"
        
        # 反応するAIがいない場合は終了
        if not ai_participants_to_respond:
            return True
        
        # AIからの応答をストリーミング生成＋並列TTS
        # 積極性レベルに基づいて発話順序を決定（積極派が先に発言）
        ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in ai_participants_to_respond]
        ai_with_activity.sort(key=lambda x: x[1], reverse=True)  # 積極性が高い順にソート
        
        for ai_name, activity_level in ai_with_activity:
            if time.time() - self.start_time > self.time_limit_minutes * 60:
                print("\n--- GD終了: 制限時間になりました ---")
                return False
            
            # 積極性レベルに基づく待ち時間（人間らしい発話タイミング）
            wait_time = self._get_speech_timing(ai_name)
            time.sleep(wait_time)
            
            # AI発言開始を通知
            if speaker_changed_signal:
                speaker_changed_signal.emit(ai_name)
            self.current_speaker = ai_name
            # ストリーミングで応答生成＋TTS並列実行
            llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task_for_ai)
            self.add_to_history(ai_name, llm_response_text)
            # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
            time.sleep(0.1)
        
        return True

    def _process_speech(self, speaker_name, speech_content, speaker_changed_signal=None, 
                       gd_thread=None, depth=0, max_depth=3):
        """
        発言を処理し、その発言に対する反応を連鎖的に処理する。
        
        Args:
            speaker_name: 発言者名（ユーザー名またはAI名）
            speech_content: 発言内容
            speaker_changed_signal: 発言者変更シグナル
            gd_thread: GDスレッドへの参照
            depth: 反応の深さ（0=元の発言、1=1回目の反応、2=2回目の反応...）
            max_depth: 最大反応深度（無限ループ防止）
        
        Returns:
            bool: 処理が成功したかどうか
        """
        # 制限時間チェック
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False
        
        # 1. 発言を履歴に追加
        self.add_to_history(speaker_name, speech_content)
        self.current_speaker = speaker_name
        
        # 2. 自己紹介の処理（ユーザー発言の場合のみ、初回のみ、SKIP_INTROがFalseの場合のみ）
        # 役割分担の処理より先に実行（自己紹介のターンでは役割分担をスキップするため）
        if speaker_name == self.username and not self.first_speech_done and not SKIP_INTRO:
            # 全AI参加者が自己紹介（固定メッセージで高速化）
            all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
            
            # TTS合成を並列実行（高速化）
            ai_introductions = {}
            with ThreadPoolExecutor(max_workers=len(all_ai_participants)) as executor:
                future_to_ai = {
                    executor.submit(self._synthesize_tts, f"{ai_name}です、よろしくお願いします。", ai_name): ai_name
                    for ai_name in all_ai_participants
                }
                
                for future in as_completed(future_to_ai):
                    ai_name = future_to_ai[future]
                    try:
                        audio_content = future.result()
                        ai_introductions[ai_name] = {
                            "text": f"{ai_name}です、よろしくお願いします。",
                            "audio": audio_content
                        }
                    except Exception as e:
                        print(f"[エラー] {ai_name}のTTS合成に失敗: {e}")
                        ai_introductions[ai_name] = {
                            "text": f"{ai_name}です、よろしくお願いします。",
                            "audio": None
                        }
            
            # 音声再生は順番に実行（重複を避けるため）
            for ai_name in all_ai_participants:
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    return False
                
                if speaker_changed_signal:
                    speaker_changed_signal.emit(ai_name)
                self.current_speaker = ai_name
                
                intro_data = ai_introductions.get(ai_name, {})
                response_text = intro_data.get("text", f"{ai_name}です、よろしくお願いします。")
                audio_content = intro_data.get("audio")
                
                # 音声再生
                if audio_content:
                    self._play_audio(audio_content)
                else:
                    # TTS合成に失敗した場合は再試行
                    self._synthesize_and_play_ai_response(response_text, ai_name)
                
                self.add_to_history(ai_name, response_text)
                time.sleep(0.2)  # 待機時間を短縮（0.3秒→0.2秒）
                # 自己紹介に対して反応しない（自己紹介は簡潔に終える）
            
            # 自己紹介完了後、本格開始アナウンス（確実に実行）
            # 音声マークを消す（システムメッセージ再生前）
            self.current_speaker = None
            if speaker_changed_signal:
                speaker_changed_signal.emit("")  # 空文字で音声マークを消す
            
            if not self.kickoff_announced and not SKIP_INTRO:
                if self.time_limit_minutes < 1:
                    time_display = f"{int(self.time_limit_minutes * 60)}秒"
                else:
                    time_display = f"{int(self.time_limit_minutes)}分"
                kickoff_message = f"今回のファシリテーターは{self.username}さんです。制限時間は{time_display}です。それでは、テーマについて議論を始めてください。"
                self.conversation_history.append({"speaker": "システム", "content": kickoff_message})
                print(f"[システム]: {kickoff_message}")
                self._synthesize_and_play_system_message(kickoff_message, gd_thread=gd_thread)
                self.kickoff_announced = True
                
                # タイマーを開始（システムメッセージ再生後に開始）
                if gd_thread:
                    gd_thread.start_timer()
            
            self.first_speech_done = True
            return True
        
        # 3. 反応者を決定
        respondents = self._decide_respondents(speaker_name, speech_content, depth=depth, max_depth=max_depth)
        
        if not respondents:
            return True  # 反応者がいない場合は正常終了
        
        # 5. 各参加者が反応
        for respondent_name in respondents:
            if time.time() - self.start_time > self.time_limit_minutes * 60:
                print("\n--- GD終了: 制限時間になりました ---")
                return False
            
            if respondent_name == self.username:
                # ユーザーが反応する場合（短いタイムアウトで待機）
                user_response = self._wait_for_user_speech(timeout=5, speaker_changed_signal=speaker_changed_signal, gd_thread=gd_thread)
                if user_response:
                    # ユーザーの反応を処理（再帰的に）
                    if not self._process_speech(self.username, user_response, speaker_changed_signal, gd_thread, depth=depth+1, max_depth=max_depth):
                        return False
            else:
                # AIが反応する場合
                # ユーザーが反応者に選ばれていない場合、AI発言前に短いタイムアウトでユーザー入力をチェック
                if self.username not in respondents:
                    # 短いタイムアウト（1秒）でユーザー入力をチェック
                    user_interruption = self._wait_for_user_speech(timeout=1.0, speaker_changed_signal=speaker_changed_signal, gd_thread=gd_thread)
                    if user_interruption:
                        # ユーザーが割り込んだ場合、その発言を優先処理
                        # 現在の連鎖反応は一旦中断し、ユーザー発言を処理
                        if not self._process_speech(self.username, user_interruption, speaker_changed_signal, gd_thread, depth=depth+1, max_depth=max_depth):
                            return False
                        # ユーザー発言の処理が終わった後、元の連鎖反応は継続せず、新しい流れに移行
                        return True
                
                # 積極性レベルに基づく待ち時間
                wait_time = self._get_speech_timing(respondent_name)
                time.sleep(wait_time)
                
                # AI発言開始を通知
                if speaker_changed_signal:
                    speaker_changed_signal.emit(respondent_name)
                self.current_speaker = respondent_name
                
                # タスクを生成
                task_for_ai = f"{speaker_name}さんの発言を踏まえ、会話履歴と文脈を理解して、ペルソナに忠実に自然に応答してください。"
                
                # ストリーミングで応答生成＋TTS並列実行
                llm_response_text = self._synthesize_and_play_ai_response_streaming(respondent_name, task_for_ai)
                self.add_to_history(respondent_name, llm_response_text)
                # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
                time.sleep(0.1)
                
                # AIの発言に対して、さらに反応（再帰的に）
                # AI発言後にもユーザー入力をチェックするため、_process_speech内でチェックされる
                if not self._process_speech(respondent_name, llm_response_text, speaker_changed_signal, gd_thread, depth=depth+1, max_depth=max_depth):
                    return False
        
        return True

    def _process_silence(self, speaker_changed_signal=None, gd_thread=None):
        """
        沈黙時（タイムアウト時）にAIが自発的に発言する。
        
        Returns:
            bool: 処理が成功したかどうか
        """
        # 制限時間チェック
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False
        
        # 積極性レベルが高いAIを選択
        all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
        ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in all_ai_participants]
        ai_with_activity.sort(key=lambda x: x[1], reverse=True)  # 積極性が高い順にソート
        
        # 積極派が自発的に発言（確率的に選択）
        ai_participants_to_respond = []
        for ai_name, activity_level in ai_with_activity:
            # 積極性レベルに基づいて発言確率を決定
            if random.random() < activity_level:
                ai_participants_to_respond.append(ai_name)
                if len(ai_participants_to_respond) >= 1:  # 1人だけ発言
                    break
        
        # 誰も発言しない場合は、積極派を強制的に選択
        if not ai_participants_to_respond and ai_with_activity:
            ai_participants_to_respond = [ai_with_activity[0][0]]  # 最も積極的なAI
        
        if not ai_participants_to_respond:
            return True
        
        task_for_ai = "会話が少し途切れているようです。会話履歴と文脈を理解して、ペルソナに忠実に自然に発言してください。議論を進めるための提案や、テーマに関する意見を述べても構いません。\n\n【重要な注意事項】\n- 発言は2〜3文程度の簡潔な長さにしてください（長すぎる発言は避けてください）\n- 役割分担の自己紹介は既に完了しているため、再度自己紹介や役割の再確認は行わないでください"
        
        # AIが自発的に発言
        for ai_name in ai_participants_to_respond:
            if time.time() - self.start_time > self.time_limit_minutes * 60:
                return False
            
            # AI発言開始を通知
            if speaker_changed_signal:
                speaker_changed_signal.emit(ai_name)
            self.current_speaker = ai_name
            llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task_for_ai)
            self.add_to_history(ai_name, llm_response_text)
            # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
            time.sleep(0.1)
            
            # AIの発言に対して、反応の連鎖処理
            if not self._process_speech(ai_name, llm_response_text, speaker_changed_signal, gd_thread, depth=0, max_depth=3):
                return False
        
        return True

    def run_conversation_loop(self, speaker_changed_signal=None, gd_thread=None):
        """
        会話ループを実行する。ターン制ではなく、発言イベントベースで進行。
        
        Args:
            speaker_changed_signal: 発言者変更シグナル
            gd_thread: GDスレッドへの参照
        
        Returns:
            bool: 処理が成功したかどうか（時間切れの場合はFalse）
        """
        # 自己紹介とキックオフをスキップする場合（テスト用）
        if SKIP_INTRO:
            print("[システム]: 自己紹介とキックオフをスキップしてGDを開始します。")
            self.first_speech_done = True
            self.kickoff_announced = True
            # タイマーを開始
            if gd_thread is not None:
                gd_thread.start_timer()
            # 会話履歴にキックオフメッセージを追加（議事録生成のため）
            if self.time_limit_minutes < 1:
                time_display = f"{int(self.time_limit_minutes * 60)}秒"
            else:
                time_display = f"{int(self.time_limit_minutes)}分"
            kickoff_message = f"今回のファシリテーターは{self.username}さんです。制限時間は{time_display}です。それでは、テーマについて議論を始めてください。"
            self.conversation_history.append({"speaker": "システム", "content": kickoff_message})
        
        # メインループ
        while self.conversation_active:
            # 制限時間チェック
            if time.time() - self.start_time > self.time_limit_minutes * 60:
                print("\n--- GD終了: 制限時間になりました ---")
                return False
            
            # ユーザー発言待ち（最初の発言のみタイムアウトなし）
            timeout = None if not self.first_speech_done else 8
            user_text = self._wait_for_user_speech(timeout=timeout, speaker_changed_signal=speaker_changed_signal, gd_thread=gd_thread)
            
            if user_text:
                # ユーザー発言を処理（反応の連鎖処理も含む）
                if not self._process_speech(self.username, user_text, speaker_changed_signal, gd_thread, depth=0, max_depth=3):
                    return False
            else:
                # 沈黙時：AIが自発的に発言（最初の発言でない場合のみ）
                if self.first_speech_done:
                    if not self._process_silence(speaker_changed_signal, gd_thread):
                        return False
                else:
                    # 最初の発言で無発話だった場合、待機
                    print("[システム]: ユーザーからの発言がありませんでした。お待ちしています。")
        
        return True

    def process_user_input(self, speaker_changed_signal=None, gd_thread=None):
        """
        ユーザーからの音声をストリーミングで認識し、GDの進行を管理する。
        AIからの応答をトリガーし、音声を再生する。
        
        Args:
            speaker_changed_signal: 発言者変更時に発火するシグナル
            gd_thread: GDスレッドへの参照（タイマー開始用）
        """
        # 制限時間チェック（エラー時のみFalseを返す）
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False

        print("\n[ユーザー]: マイク入力待ち...")
        
        # 待機中状態を表示
        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
            self.gui_window.gd_screen.show_waiting()
        
        # 音声入力開始時のコールバック
        def on_speaking_start():
            if speaker_changed_signal:
                speaker_changed_signal.emit(self.username)
            self.current_speaker = self.username
            # ユーザーが話している状態を表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.show_user_speaking(self.username)

        client = speech.SpeechClient()
        
        # PyAudioインスタンスを再利用
        p_audio = self._get_p_audio()
        
        language_code = "ja-JP"
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        user_text = ""
        try:
            # GD開始後の最初の発言の場合のみ、タイムアウトを適用しない
            current_timeout = None if not self.first_speech_done else 8
            # MicrophoneStreamにp_audioとspeaking_callbackを渡す
            with MicrophoneStream(RATE, CHUNK, timeout=current_timeout, p_audio=p_audio, speaking_callback=on_speaking_start) as stream:
                audio_generator = stream.generator()
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = client.streaming_recognize(streaming_config, requests)
                print("認識中...")
                for response in responses:
                    # 時間チェック（音声認識中）
                    if time.time() - self.start_time > self.time_limit_minutes * 60:
                        print("\n--- GD終了: 制限時間になりました（音声認識中） ---")
                        self.conversation_active = False
                        return ""
                    
                    if not response.results or not response.results[0].alternatives:
                        continue
                    result = response.results[0]
                    if result.is_final:
                        user_text = result.alternatives[0].transcript.strip()
                        print(f"\n[あなた]: {user_text}")
                        break
        except Exception as e:
            print(f"音声認識エラーが発生しました: {e}")
            # エラー時も待機中に戻す
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.show_waiting()
            return False
        finally:
            # p_audio.terminate() を削除（再利用のため）
            pass

        # ユーザーの発言がなかった場合の処理
        if not user_text or user_text is None:
            # GD開始後の最初の発言で無発話だった場合、システムが何もせずに待機する
            if not self.first_speech_done:
                print("[システム]: ユーザーからの発言がありませんでした。お待ちしています。")
                # 待機中状態を表示
                if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                    self.gui_window.gd_screen.show_waiting()
                return True # ユーザーが発話するまでこの関数を再度呼び出す
            
            # GD進行中に無発話だった場合（沈黙時の自発的な発言）
            # 積極派が自発的に話し出す（外崎・伊藤の研究に基づく）
            all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
            # 積極性レベルが高いAI（積極派）を優先的に選択
            ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in all_ai_participants]
            ai_with_activity.sort(key=lambda x: x[1], reverse=True)  # 積極性が高い順にソート
            
            # 積極派が自発的に発言（確率的に選択）
            ai_participants_to_respond = []
            for ai_name, activity_level in ai_with_activity:
                # 積極性レベルに基づいて発言確率を決定
                if random.random() < activity_level:
                    ai_participants_to_respond.append(ai_name)
                    if len(ai_participants_to_respond) >= 1:  # 1人だけ発言
                        break
            
            # 誰も発言しない場合は、積極派を強制的に選択
            if not ai_participants_to_respond and ai_with_activity:
                ai_participants_to_respond = [ai_with_activity[0][0]]  # 最も積極的なAI
            
            task_for_ai = "会話が少し途切れているようです。会話履歴と文脈を理解して、ペルソナに忠実に自然に発言してください。議論を進めるための提案や、テーマに関する意見を述べても構いません。\n\n【重要な注意事項】\n- 発言は2〜3文程度の簡潔な長さにしてください（長すぎる発言は避けてください）\n- 役割分担の自己紹介は既に完了しているため、再度自己紹介や役割の再確認は行わないでください"
            # AI応答のループ（ストリーミング版を使用）
            for ai_name in ai_participants_to_respond:
                # AI発言開始を通知
                if speaker_changed_signal:
                    speaker_changed_signal.emit(ai_name)
                self.current_speaker = ai_name
                llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task_for_ai)
                self.add_to_history(ai_name, llm_response_text)
                # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
                time.sleep(0.1)
                
                # AIが発言した後、他のAIが反応する（自然な会話の流れを維持）
                if not self._process_ai_response_to_speech(ai_name, llm_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                    # 時間切れの場合は終了
                    if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                        self.gui_window.gd_screen.show_waiting()
                    return False
            # 処理完了後、待機中に戻す
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.show_waiting()
            return True

        # ユーザーの発言があった場合の処理
        self.add_to_history(self.username, user_text)
        self.current_speaker = self.username
        
        ai_participants_to_respond = []
        task_for_ai = ""

        if not self.first_speech_done:
            all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
            ai_participants_to_respond = all_ai_participants
            task_for_ai = "ファシリテーターが自己紹介を終えました。あなたも「（名前）です、よろしくお願いします」のように、名前と一言だけで簡潔に自己紹介してください。ペルソナや意気込みは述べず、挨拶だけにしてください。"
        else:
            # 特定の参加者への質問かどうかをチェック（名前が含まれている場合）
            target_ai_id = self._get_target_ai_from_text(user_text)
            if target_ai_id:
                # 指名されたAIが応答
                ai_participants_to_respond.append(target_ai_id)
                task_for_ai = f"ファシリテーターがあなた（{target_ai_id}）に話しかけています。会話履歴と文脈を理解して、自然に応答してください。"
            else:
                # 一般的な発言の場合、AIが自発的に判断して応答
                all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
                
                # 積極性レベルに基づいて自発的に応答
                ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in all_ai_participants]
                ai_with_activity.sort(key=lambda x: x[1], reverse=True)
                
                # 積極性レベルに基づいて発言確率を決定
                for ai_name, activity_level in ai_with_activity:
                    if random.random() < activity_level * 0.7:  # 積極性に基づく発言確率
                        ai_participants_to_respond.append(ai_name)
                        if len(ai_participants_to_respond) >= 2:  # 最大2人
                            break
                
                # 誰も発言しない場合は、積極派を強制的に選択
                if not ai_participants_to_respond and ai_with_activity:
                    ai_participants_to_respond = [ai_with_activity[0][0]]  # 最も積極的なAI
                
                task_for_ai = "ファシリテーターの発言を踏まえ、会話履歴と文脈を理解して、ペルソナに忠実に自然に応答してください。"

        # AIからの応答をストリーミング生成＋並列TTS（最高速化）
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False
        
        # 各AI参加者を処理
        # 自己紹介時はLLM応答生成を並列化して高速化
        if not self.first_speech_done and len(ai_participants_to_respond) > 2:
            print(f"[システム]: {len(ai_participants_to_respond)}名のAI自己紹介を並列生成中...")
            
            # 並列生成開始時に「考えている」状態を表示
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.show_ai_thinking(f"{len(ai_participants_to_respond)}名のAI")
            
            # LLM応答を並列生成
            ai_responses = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_ai = {
                    executor.submit(self._get_ai_response, ai_name, task_for_ai): ai_name 
                    for ai_name in ai_participants_to_respond
                }
                
                for future in as_completed(future_to_ai):
                    ai_name = future_to_ai[future]
                    try:
                        response_text = future.result()
                        ai_responses[ai_name] = response_text
                        print(f"[システム]: {ai_name}の応答生成完了")
                    except Exception as e:
                        print(f"[エラー] {ai_name}の応答生成に失敗: {e}")
                        ai_responses[ai_name] = f"{ai_name}です、よろしくお願いします。"
            
            # 並列生成完了後、待機中に戻す
            if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                self.gui_window.gd_screen.show_waiting()
            
            # 音声再生は順番に実行（元の順序を維持）
            last_ai_name = None
            last_response_text = None
            for ai_name in ai_participants_to_respond:
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました ---")
                    return False
                
                # AI発言開始を通知
                if speaker_changed_signal:
                    speaker_changed_signal.emit(ai_name)
                self.current_speaker = ai_name
                
                # 生成済みの応答を音声再生
                response_text = ai_responses.get(ai_name, "")
                if response_text:
                    self._synthesize_and_play_ai_response(response_text, ai_name)
                    self.add_to_history(ai_name, response_text)
                    last_ai_name = ai_name
                    last_response_text = response_text
                # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
                time.sleep(0.1)
            
            # 自己紹介完了後、最後のAIの発言に対して他のAIが反応する
            if last_ai_name and last_response_text:
                if not self._process_ai_response_to_speech(last_ai_name, last_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                    return False  # 時間切れの場合は終了
            
            # 自己紹介完了後、本格開始アナウンス
            if not self.kickoff_announced:
                # 時間表示を調整（10秒未満の場合は秒数で表示）
                if self.time_limit_minutes < 1:
                    time_display = f"{int(self.time_limit_minutes * 60)}秒"
                else:
                    time_display = f"{int(self.time_limit_minutes)}分"
                kickoff_message = f"今回のファシリテーターは{self.username}さんです。制限時間は{time_display}です。それでは、テーマについて議論を始めてください。"
                self.conversation_history.append({"speaker": "システム", "content": kickoff_message})
                print(f"[システム]: {kickoff_message}")
                self._synthesize_and_play_system_message(kickoff_message, gd_thread=gd_thread)
                self.kickoff_announced = True
                
                # タイマーを開始（システムメッセージ再生後に開始）
                if gd_thread:
                    gd_thread.start_timer()
        else:
            # 通常時は順次処理（ストリーミング＋TTS並列で高速化）
            # 積極性レベルに基づいて発話順序を決定（積極派が先に発言）
            ai_with_activity = [(ai, self.participants[ai].get("activity_level", 0.5)) for ai in ai_participants_to_respond]
            ai_with_activity.sort(key=lambda x: x[1], reverse=True)  # 積極性が高い順にソート
            
            for ai_name, activity_level in ai_with_activity:
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました ---")
                    if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                        self.gui_window.gd_screen.show_waiting()
                    return False
                
                # 積極性レベルに基づく待ち時間（人間らしい発話タイミング）
                wait_time = self._get_speech_timing(ai_name)
                time.sleep(wait_time)
                
                # AI発言開始を通知
                if speaker_changed_signal:
                    speaker_changed_signal.emit(ai_name)
                self.current_speaker = ai_name
                # ストリーミングで応答生成＋TTS並列実行
                llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task_for_ai)
                self.add_to_history(ai_name, llm_response_text)
                # 再生終了後の待ち時間を短縮（0.3秒→0.1秒）
                time.sleep(0.1)
                
                # AIが発言した後、他のAIが反応する（自己紹介や役割分担の場合はスキップ）
                # 自己紹介や役割分担以外の場合のみ、他のAIが反応する
                if self.first_speech_done:
                    if not self._process_ai_response_to_speech(ai_name, llm_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                        # 時間切れの場合は終了
                        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
                            self.gui_window.gd_screen.show_waiting()
                        return False

        # 処理完了後、待機中に戻す
        if self.gui_window and hasattr(self.gui_window, 'gd_screen'):
            self.gui_window.gd_screen.show_waiting()
        
        return True

    def _has_recorder(self) -> bool:
        """
        書記が任命されているかどうかを確認する
        
        Returns:
            bool: 書記が任命されていればTrue、そうでなければFalse
        """
        for ai_name, info in self.participants.items():
            if info.get("assigned_role") == "書記":
                return True
        return False
    
    def _user_mentioned_role_assignment(self, recent_turns=5) -> bool:
        """
        直近の会話履歴でユーザーが役割分担について言及しているかを確認する
        
        Args:
            recent_turns: 確認する直近の発言数
        
        Returns:
            bool: ユーザーが役割分担について言及していればTrue、そうでなければFalse
        """
        # 直近の発言を取得
        recent_history = self.conversation_history[-recent_turns:] if len(self.conversation_history) > recent_turns else self.conversation_history
        
        # ユーザーの発言をチェック
        role_assignment_keywords = [
            "役割分担", "役割を", "役割に", "役割が",
            "書記を", "書記に", "書記が", "書記と", "書記で",
            "タイムキーパーを", "タイムキーパーに", "タイムキーパーが",
            "書記を担当", "タイムキーパーを担当",
            "書記お願い", "書記をお願い", "書記をお願いします",
            "タイムキーパーお願い", "タイムキーパーをお願い",
            "記録を", "記録を担当", "議事録を", "議事録を記録",
            "役割を決め", "役割を決めましょう", "役割を決めます"
        ]
        
        for msg in recent_history:
            if msg['speaker'] == self.username:
                user_text_lower = msg['content'].lower()
                if any(keyword in user_text_lower for keyword in role_assignment_keywords):
                    return True
        
        return False
    
    def get_minutes_text(self) -> str:
        """
        ファシリテーション向けの構造化議事録を生成して返す。
        LLM（Gemini）が会話履歴を分析し、決定事項・主要意見・論点を自動抽出。
        
        注意: 書記が任命されていない場合は、議事録を生成せずにメッセージを返す。
        """
        # 1. 書記が任命されていない場合は空文字列を返す（議事録を表示しない）
        if not self._has_recorder():
            return ""
        
        # 2. LLMに会話履歴を渡して構造化議事録を生成
        # 注意: conversation_historyはGD開始時からの全発言を含むため、
        # 書記が任命された後でも、それまでの会話も含めて議事録が生成される
        conversation_text = "\n".join([
            f"[{msg['speaker']}] {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        prompt = f"""以下のグループディスカッションの会話履歴を分析し、ファシリテーター向けの議事録を作成してください。

【会話履歴】
{conversation_text}

【参加者情報】
ファシリテーター: {self.username}
その他の参加者: {', '.join([name for name in self.participants.keys() if name != self.username])}

【出力形式】
以下の形式で出力してください:

【決定事項】✅
• （合意に至った内容を箇条書き、なければ「なし」）

【主要意見】💡
[{self.username}]
  - （{self.username}さんの重要な提案や視点）
[その他の参加者名]
  - （その人の重要な提案や視点）
（各参加者ごとに記載、システムメッセージは除外。参加者名は実際の名前を使用すること）

簡潔に、要点のみを記載してください。全発言を列挙するのではなく、重要なポイントのみ抽出してください。"""

        try:
            messages = [
                {"role": "user", "parts": [prompt]}
            ]
            response = self.gemini_model.generate_content(messages)
            
            if not response._result.candidates:
                structured_minutes = "（議事録の生成に失敗しました）"
            else:
                structured_minutes = response.text.strip()
        except Exception as e:
            print(f"議事録生成エラー: {e}")
            structured_minutes = "（議事録の生成中にエラーが発生しました）"
        
        # 2. 構造化議事録を返す
        return structured_minutes

# --- スクリプトの実行（GDManagerの動作確認用） ---
if __name__ == "__main__":
    print("--- GDManagerの統合テスト開始 ---") 
    # 環境変数が設定されているか最終チェック (dotenvが読み込む)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_API_KEY"): # GOOGLE_API_KEYもチェック
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY) が設定されていません。")
        exit()

    import signal
    # Ctrl+C を QEvent ではなくプロセスデフォルトの割り込みに戻す（PySideのイベントループでも Ctrl+C を受け取れるようにする）
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Qtの警告メッセージを抑制（オプション: 必要に応じてコメントアウト）
    from PySide6.QtCore import Qt, qInstallMessageHandler
    
    def qt_message_handler(msg_type, msg_context, msg_string):
        """Qtの警告メッセージをフィルタリング"""
        # setGeometry関連の警告を無視
        if "setGeometry" in msg_string:
            return
        # その他の警告は通常通り表示
        if msg_type == Qt.QtMsgType.QtWarningMsg:
            print(f"Qt警告: {msg_string}")
    
    qInstallMessageHandler(qt_message_handler)

    # GUIアプリケーションのインスタンスを作成
    app = QApplication(sys.argv)

    # GUIウィンドウを作成
    gui_window = GDReportWindow()
    
    # GDManagerとスレッドを保持するコンテナ
    class GDContext:
        def __init__(self):
            self.manager = None
            self.gd_thread = None
    
    context = GDContext()
    
    def on_start_gd_with_username_and_round(username, round_number):
        """GD開始時にGDManagerとスレッドを作成"""
        try:
            # 固定テーマを取得
            theme = get_fixed_gd_theme(round_number)
            
            # GDManagerをユーザー名と固定テーマで作成
            context.manager = GDManager(
                gui_window, 
                username=username, 
                gd_theme=theme,
                num_ai_participants=3
            )
            gui_window.set_manager(context.manager)
            
            # --- GUIの初期表示（メインスレッドで実行） ---
            # GD画面に遷移（自動開始時も含む）
            gui_window.stacked_widget.setCurrentIndex(4)  # GD画面へ
            QApplication.processEvents()  # UI更新を確実に実行
            
            # テーマタイトルを画面・議事録エリアに表示
            gui_window.set_theme(theme)
            gui_window.gd_screen.set_theme(theme)
            # ユーザーの役職をGUIに反映
            gui_window.gd_screen.update_participant_role(username, "ファシリテーター")
            # タイマーの初期値を設定
            initial_minutes = int(context.manager.time_limit_minutes)
            initial_seconds = int(round((context.manager.time_limit_minutes - initial_minutes) * 60))
            gui_window.gd_screen.update_timer(initial_minutes, initial_seconds)
            # 議事録をクリア（前回のGDの議事録が残らないようにする）
            gui_window.set_minutes("")
            
            # まずローディングを消し、GD画面を完全に表示してからシステム発話を行う
            gui_window.gd_screen.hide_loading()
            QApplication.processEvents()  # UI更新を確実に実行
            
            # GD開始時の初期メッセージ（自己紹介の案内など）を再生
            # ※この処理中はTTSでメインスレッドがブロックされるが、すでにローディングは消えている
            # SKIP_INTROがTrueの場合はスキップ
            if not SKIP_INTRO:
                context.manager._initialize_gd()
            else:
                # SKIP_INTROがTrueの場合でも、会話履歴には追加（議事録生成のため）
                theme_title = context.manager.gd_theme.splitlines()[0] if context.manager.gd_theme else ""
                initial_message = (
                    f"グループディスカッションを始めます。本日のテーマは『{theme_title}』です。"
                    f"まず、{context.manager.username}さんから順番に、お名前と一言だけ簡単に自己紹介をお願いします。"
                )
                context.manager.conversation_history.append({"speaker": "システム", "content": initial_message})
            
            # GDThreadを作成（ラウンド番号と実験群/統制群を渡す）
            context.gd_thread = GDThread(
                context.manager, 
                round_number=round_number,
                experiment_group=gui_window.experiment_group
            )
            
            # シグナル接続
            if round_number == 1:
                # 1回目: フィードバックを表示
                context.gd_thread.finished.connect(gui_window.set_feedback)
            else:
                # 2回目: 実験終了ポップアップを表示
                context.gd_thread.finished.connect(gui_window._on_second_gd_finished)
            
            context.gd_thread.minutes_updated.connect(gui_window.set_minutes)
            context.gd_thread.speaker_changed.connect(gui_window.update_speaker)
            context.gd_thread.timer_updated.connect(gui_window.update_timer)
            # 役割更新（メインスレッドで参加者ラベルを更新）
            context.gd_thread.role_updated.connect(gui_window.gd_screen.update_participant_role)
            # システム発話中シグナル接続
            context.gd_thread.system_speaking.connect(lambda is_speaking: gui_window.show_system_speaking() if is_speaking else gui_window.hide_system_speaking())
            # フィードバック生成進捗表示（1回目のみ）
            if round_number == 1:
                context.gd_thread.feedback_progress.connect(gui_window._on_feedback_progress)
            
            # スレッド開始
            context.gd_thread.start()
            context.gd_thread.start_gd()
        except Exception as e:
            import traceback
            error_msg = f"GD開始時にエラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            # エラー時もローディングを非表示
            gui_window.gd_screen.hide_loading()
            QMessageBox.critical(
                gui_window,
                "エラー",
                f"GD開始時にエラーが発生しました:\n{str(e)}\n\n詳細はコンソールを確認してください。"
            )
            # エラーが発生した場合は、GD開始確認画面に戻る
            gui_window.stacked_widget.setCurrentIndex(2)
    
    # GUIからのシグナルを接続
    gui_window.start_gd_requested.connect(on_start_gd_with_username_and_round)

    # GUIウィンドウを表示
    gui_window.show()
    
    # テスト用: 自動的にGDを開始（会話部分のテスト用）
    # 開発モードでも最初から最後までストーリーを確認するため、自動開始は無効化
    # if DEV_MODE:
    #     print(f"[テストモード] 自動的にGDを開始します（ユーザー名: {DEV_DEFAULT_USERNAME}, ラウンド: 1）")
    #     # 少し遅延させてGUI初期化を待つ
    #     from PySide6.QtCore import QTimer
    #     QTimer.singleShot(1000, lambda: on_start_gd_with_username_and_round(DEV_DEFAULT_USERNAME, 1))

    # アプリケーション終了時にスレッド停止と履歴リセットを行うハンドラを登録
    def _on_about_to_quit():
        try:
            print("アプリ終了: GDスレッド停止中...")
            if context.gd_thread:
                context.gd_thread.stop()
                context.gd_thread.quit()
                context.gd_thread.wait(2000)  # 最大2秒待機
            
            # 使用済みテーマの履歴をリセット
            print("アプリ終了: テーマ履歴をリセット中...")
            reset_used_themes()
        except Exception as e:
            print(f"終了処理エラー: {e}")

    app.aboutToQuit.connect(_on_about_to_quit)

    # GUIアプリケーションのイベントループを開始
    sys.exit(app.exec())