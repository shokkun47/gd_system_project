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
    GroupSelectionScreen, GDStartConfirmScreen, ControlGroupAfterFirstScreen
)

# --- システム共通設定 ---
# gd_manager.py は src/ ディレクトリにあるため、プロジェクトルートパスを正確に取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 開発モード設定 ===
# 開発中は True、本番は False に設定
DEV_MODE = True
DEV_DEFAULT_USERNAME = "テスト"  # 開発モード時のデフォルト名字
DEV_THINKING_SECONDS = 0  # 開発モード時の思考時間（0=スキップ）
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

# 実験用固定テーマ
THEME_ROUND_1 = """学園祭模擬店の売上向上施策

あなたはゼミの模擬店リーダーです。昨年の「焼きそば」は売上が伸び悩み、利益がほとんど出ませんでした。今年の学園祭では「利益を昨年の1.5倍にする」ことが目標です。これから15分間で、メンバー（AI）と議論し、目標達成のための「具体的な施策を1つ」決定してください。※予算には限りがあります。"""

THEME_ROUND_2 = """オープンキャンパスの来場者数増加施策

あなたはオープンキャンパスの学生リーダーです。近年、高校生の来場者数が減少傾向にあり、大学側から対策を求められています。今年の開催に向けて、「来場者を確実に増やすための目玉企画」を1つ決定してください。これから15分間で、メンバー（AI）と議論し、結論を出してください。※学生スタッフだけで実施できる内容に限ります。"""

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
                    print(f"\n[システム]: 無音時間が{self._timeout}秒続いたため、入力を終了します。")

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
        self.user_input_screen = UserInputScreen()
        self.gd_start_confirm_screen = GDStartConfirmScreen()
        self.gd_screen = GDScreen()
        self.feedback_screen = FeedbackScreen()
        self.control_after_first_screen = ControlGroupAfterFirstScreen()
        
        # 画面を追加（インデックスを定義）
        self.stacked_widget.addWidget(self.group_selection_screen)      # index 0
        self.stacked_widget.addWidget(self.user_input_screen)           # index 1
        self.stacked_widget.addWidget(self.gd_start_confirm_screen)     # index 2
        self.stacked_widget.addWidget(self.gd_screen)                   # index 3
        self.stacked_widget.addWidget(self.feedback_screen)             # index 4
        self.stacked_widget.addWidget(self.control_after_first_screen)  # index 5
        
        # シグナル接続
        self.group_selection_screen.group_selected.connect(self._on_group_selected)
        self.user_input_screen.system_start_requested.connect(self._on_username_entered)
        self.gd_start_confirm_screen.confirmed.connect(self._on_gd_start_confirmed)
        self.gd_start_confirm_screen.cancelled.connect(self._on_gd_start_cancelled)
        self.feedback_screen.next_gd_requested.connect(self._on_next_gd_requested)
        self.control_after_first_screen.next_gd_requested.connect(self._on_next_gd_requested)
        
        # 状態管理
        self.current_username = ""  # 名字のみ（GD内で使用）
        self.current_fullname = ""  # フルネーム（保存時に使用）
        self.experiment_group = None  # "experimental" または "control"
        self.current_gd_round = 1  # 1 または 2
        self.first_gd_feedback = {}  # 1回目のフィードバック（実験群のみ）
        
        # 初期画面を表示
        self.stacked_widget.setCurrentIndex(0)  # 実験群/統制群選択画面
    
    def _on_group_selected(self, group):
        """実験群/統制群選択 → ユーザー名入力画面へ"""
        self.experiment_group = group
        self.stacked_widget.setCurrentIndex(1)  # ユーザー名入力画面へ
    
    def _on_username_entered(self, lastname, firstname):
        """ユーザー名入力 → 1回目GD開始確認画面へ"""
        self.current_username = lastname  # 名字のみ（GD内で使用）
        self.current_fullname = lastname + firstname  # フルネーム（保存時に使用）
        self.current_gd_round = 1
        # 1回目GD開始確認画面を表示
        self.gd_start_confirm_screen.set_message(
            "これから1回目のグループディスカッションを開始します。\n準備はよろしいですか？"
        )
        self.stacked_widget.setCurrentIndex(2)  # GD開始確認画面へ
    
    def _on_gd_start_confirmed(self):
        """GD開始確認 → GD進行画面へ"""
        # 画面遷移を確実にする
        self.stacked_widget.setCurrentIndex(3)  # GD画面へ
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
            self.stacked_widget.setCurrentIndex(1)  # ユーザー名入力画面へ戻る
        else:
            # 2回目: フィードバック画面または統制群用画面へ戻る
            if self.experiment_group == "experimental":
                self.stacked_widget.setCurrentIndex(4)  # フィードバック画面へ
            else:
                self.stacked_widget.setCurrentIndex(5)  # 統制群用画面へ
    
    def _on_next_gd_requested(self):
        """2回目GD開始要求 → 2回目GD開始確認画面へ"""
        self.current_gd_round = 2
        # 2回目GD開始確認画面を表示
        self.gd_start_confirm_screen.set_message(
            "これから2回目のグループディスカッションを開始します。\n準備はよろしいですか？"
        )
        self.stacked_widget.setCurrentIndex(2)  # GD開始確認画面へ
    
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
            # 実験群: フィードバック画面を表示（2回目開始ボタン付き）
            self.feedback_screen.set_feedback(feedback_dict, show_next_button=True)
            self.stacked_widget.setCurrentIndex(4)  # フィードバック画面へ
        else:
            # 統制群: 学習用ドキュメント画面へ（CSVは既に保存済み）
            self.stacked_widget.setCurrentIndex(5)  # 統制群用画面へ
    
    def _on_feedback_progress(self, message):
        """フィードバック生成進捗を表示"""
        # フィードバック画面に切り替えて進捗を表示
        self.stacked_widget.setCurrentIndex(4)  # フィードバック画面へ
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
        # 自動保存（2回目）
        if hasattr(self, 'manager') and self.manager:
            try:
                filepath = self.manager.save_feedback_report(
                    feedback_dict, 
                    self.current_fullname,  # フルネームで保存
                    round_number=2,
                    experiment_group=self.experiment_group
                )
                print(f"[システム]: 2回目のフィードバックを自動保存しました: {filepath}")
            except Exception as e:
                print(f"[警告]: 2回目のフィードバックの自動保存に失敗しました: {e}")
        
        # 実験終了ポップアップを表示
        msg = QMessageBox(self)
        msg.setWindowTitle("実験終了")
        msg.setText("実験お疲れさまでした")
        msg.setInformativeText("2回のグループディスカッションが完了しました。\nご協力ありがとうございました。")
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
        """外部からタイマー開始を要求する"""
        if not self._timer_started:
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
        
        running_gd = True
        while self._running and running_gd:
            try:
                # AI応答を実行（並列化済み）
                # 発言者の切り替えはprocess_user_input内で行う
                running_gd = self.manager.process_user_input(self.speaker_changed, gd_thread=self)
                
                # 議事録生成を別スレッドで非同期実行（UI応答性向上）
                def update_minutes_async():
                    try:
                        minutes_text = self.manager.get_minutes_text()
                        self.minutes_updated.emit(minutes_text)
                    except Exception as e:
                        print(f"議事録生成エラー: {e}")
                        self.minutes_updated.emit(f"議事録生成エラー: {e}")
                
                minutes_thread = threading.Thread(target=update_minutes_async, daemon=True)
                minutes_thread.start()

                if not running_gd:
                    break
            except KeyboardInterrupt:
                # Ctrl+C が来た場合はループを抜ける
                break
            except Exception as e:
                print(f"GDThreadエラー: {e}")
                break
        
        # GD終了後にシステムメッセージを音声で通知
        if self.round_number == 1:
            # 1回目: フィードバックを生成（進捗表示付き）
            if self.experiment_group == "experimental":
                end_message = "グループディスカッションを終了します。お疲れ様でした。フィードバックレポートを生成します。"
            else:
                end_message = "グループディスカッションを終了します。お疲れ様でした。"
            self.manager._synthesize_and_play_system_message(end_message)
            
            # フィードバックを生成し、シグナルで送信（進捗表示付き、統制群の場合はAIフィードバックは生成しない）
            report = self.manager.generate_simple_feedback_report(
                progress_signal=self.feedback_progress if self.experiment_group == "experimental" else None,
                experiment_group=self.experiment_group
            )
            self.finished.emit(report)
        else:
            # 2回目: フィードバックを生成するが表示はしない（データは自動保存される）
            end_message = "グループディスカッションを終了します。お疲れ様でした。"
            self.manager._synthesize_and_play_system_message(end_message)
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
        
        # タイマー開始時刻を記録
        timer_start_time = time.time()
        
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
        self.time_limit_minutes = 10 / 60  # 動作確認用: 10秒（通常は15分）
        self.start_time = time.time() 
        self.turn_count = 0 
        self.current_speaker = "システム" 
        self.roles_assigned = False
        self.kickoff_announced = False  # 本格開始アナウンス済みフラグ
        
        # --- APIクライアントの初期化 ---
        try:
            # self.speech_client = speech.SpeechClient()  # ASRクライアント
            # self.tts_client = texttospeech.TextToSpeechClient() # TTSクライアント
            
            # --- Gemini APIの認証とクライアント初期化 ---
            # .envファイルから GOOGLE_API_KEY 環境変数を読み込みます
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # <-- GOOGLE_API_KEY を .env に設定すること
            # GD中・採点用モデル（高速）
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            # フィードバック生成専用モデル（高精度）
            try:
                self.gemini_feedback_model = genai.GenerativeModel(GEMINI_FEEDBACK_MODEL)
            except Exception as e:
                # フィードバック専用モデルの初期化に失敗した場合は、通常モデルで代用
                print(f"フィードバック専用モデル({GEMINI_FEEDBACK_MODEL})の初期化に失敗しました。通常モデルで代用します: {e}")
                self.gemini_feedback_model = self.gemini_model
            print("全APIクライアントを GDManager 内で初期化しました。")
        except Exception as e:
            error_msg = f"エラー: APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}\nヒント: GOOGLE_APPLICATION_CREDENTIALS と GOOGLE_API_KEY が正しく設定されていますか？"
            print(error_msg)
            raise RuntimeError(error_msg)  # exit()の代わりに例外を投げる
        
        # self.p_audio = pyaudio.PyAudio() 
        
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
            # 積極派: 0.8, 慎重派: 0.5, 消極派: 0.2
            activity_level = {"積極派": 0.8, "慎重派": 0.5, "消極派": 0.2}[persona_type]
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
- 積極派の意見に対して現実的な懸念を示す
- リスクを気にする
- 実現可能性を重視する
- 批判的な視点から問題点を指摘する
- 議論を深掘りすることを好む

【発言スタイル】
- 積極派の後に発言することが多い
- 発言の長さは中程度
- 「確かにそうですが、〜という懸念があります」「リスクとして〜」といった表現を使う
- 慎重に検討する姿勢を示す

【注意事項】
- 文脈（現在のフェーズや議論の流れ）を理解して発言すること
- 否定的にならず、建設的な批判を心がけること"""
        
        else:  # 消極派
            return f"""あなたはGDの参加者である{ai_id}です。以下の特徴を持っています：

【性格・行動パターン】
- 自分からは発言しない
- 指名されたら短く答える
- 「特にないです」「そうですね」と言いがち
- 発言が短い
- 他の参加者の意見に同調することが多い

【発言スタイル】
- 指名された時のみ発言する傾向がある
- 発言の長さは短め
- 「そうですね」「確かに」「特にないです」といった短い応答が多い
- 自発的な発言はほとんどしない

【注意事項】
- 文脈（現在のフェーズや議論の流れ）を理解して発言すること
- 完全に沈黙するのではなく、最低限の反応は示すこと"""
    
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
            f"あなたはGDの参加者である{ai_id}です。\n\n"
            f"【あなたのペルソナ】\n"
            f"{persona_info['persona']}\n\n"
            f"【重要な指示】\n"
            f"- これまでの会話履歴をよく読み、文脈を理解して発言してください\n"
            f"- 直前の発言（特に他の参加者の発言）を踏まえて、自然な会話の流れを作ってください\n"
            f"- ペルソナに忠実に、人間らしい自然な発言を心がけてください\n"
            f"- 「です・ます」調で話してください\n"
            f"- 長すぎず短すぎず、適切な長さで発言してください（ペルソナタイプに応じて調整）\n"
            f"- AIであることを示すような発言は避けてください\n"
            f"- 会話の流れに沿って、適度に感情や反応を示してください\n"
        )
        messages_content.append({"role": "user", "parts": [system_instruction_content]})
        messages_content.append({"role": "model", "parts": ["了解しました。"]}) # AIの初期応答をシミュレート
        
        # 過去の会話履歴（直近の会話を優先的に含める）
        if include_current_history:
            # 直近10ターン分の会話履歴を含める（文脈理解のため）
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
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
        
        messages_content.append({"role": "user", "parts": [
            f"【状況】\n"
            f"残り時間: {remaining_minutes}分\n"
            f"あなたの役割: {self.participants[ai_id].get('assigned_role', 'なし')}\n\n"
            f"【あなたへの指示】\n"
            f"{task_instruction}\n\n"
            f"上記の指示に従い、ペルソナに忠実に、自然な日本語で発言を生成してください。"
        ]})
    
        return messages_content
    
    def _get_ai_response_streaming(self, ai_id, task_for_ai, include_current_history=True):
        """
        特定のAI参加者（ai_id）の発言をストリーミングで生成し、
        文が完成するごとにyieldする（TTS並列実行用）
        
        Yields:
            str: 生成された文（句点で区切られた単位）
        """
        messages = self._generate_ai_prompt(ai_id, task_for_ai, include_current_history)
        print(f" (GDマネージャー -> Geminiへの指示(ストリーミング) for {ai_id}): {task_for_ai[:80]}...")

        try:
            # ストリーミングモードで生成
            response_stream = self.gemini_model.generate_content(messages, stream=True)
            
            buffer = ""
            for chunk in response_stream:
                if chunk.text:
                    buffer += chunk.text
                    # 文の区切り（。！？）を検出
                    while any(delimiter in buffer for delimiter in ['。', '！', '？', '\n']):
                        for delimiter in ['。', '！', '？', '\n']:
                            if delimiter in buffer:
                                idx = buffer.index(delimiter)
                                sentence = buffer[:idx+1].strip()
                                buffer = buffer[idx+1:]
                                if sentence:
                                    yield sentence
                                break
            
            # 残りのバッファも返す
            if buffer.strip():
                yield buffer.strip()
                
        except Exception as e:
            print(f"エラー: Geminiストリーミング中にエラーが発生しました: {e}")
            yield f"（{ai_id}）応答エラーのため発言できません。"
    
    def _get_ai_response(self, ai_id, task_for_ai, include_current_history=True):
        """
        特定のAI参加者（ai_id）の発言をLLM（Gemini）に生成させる。
        返り値は常に文字列（空のときはフォールバックテキスト）にする。
        ストリーミング版を使用して全文を結合。
        """
        full_response = ""
        for sentence in self._get_ai_response_streaming(ai_id, task_for_ai, include_current_history):
            full_response += sentence
        return full_response if full_response else f"（{ai_id}）応答が空です。"

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
    
    def _play_audio(self, audio_content):
        """音声データを再生"""
        p_audio = pyaudio.PyAudio()
        stream = None 
        try:
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True
            )
            stream.write(audio_content) 
        except Exception as e:
            print(f"音声再生エラー: {e}")
        finally:
            if stream: 
                stream.stop_stream()
                stream.close()
            p_audio.terminate()
    
    def _synthesize_and_play_ai_response_streaming(self, ai_id, task_for_ai):
        """
        ストリーミングでLLM応答を生成し、文ごとにTTS→再生を並列実行。
        最初の文が出たらすぐに再生開始（体感速度が大幅向上）
        
        Returns:
            str: 全応答テキスト
        """
        full_text = ""
        audio_queue = queue.Queue()
        tts_futures = []
        
        # TTS並列実行用のExecutor
        tts_executor = ThreadPoolExecutor(max_workers=3)
        
        print(f"[{ai_id}]: ストリーミング応答生成中...")
        
        # ストリーミングでLLM応答を生成し、文ごとにTTS送信
        for sentence in self._get_ai_response_streaming(ai_id, task_for_ai):
            full_text += sentence
            # 各文をTTSに並列送信
            future = tts_executor.submit(self._synthesize_tts, sentence, ai_id)
            tts_futures.append(future)
        
        print(f"[{ai_id}]: {full_text}")
        
        # TTS結果を順次取得して再生
        print(f"[{ai_id}]: 音声を再生中...")
        for future in tts_futures:
            audio_content = future.result()
            if audio_content:
                self._play_audio(audio_content)
        
        tts_executor.shutdown(wait=False)
        print(f"[{ai_id}]: 再生終了。")
        
        return full_text
    
    def _synthesize_and_play_ai_response(self, text_to_synthesize, ai_id):
        """AIの応答テキストを音声合成し、再生する。（非ストリーミング版、互換性用）"""
        if not text_to_synthesize:
            print(f"[{ai_id}]: 合成対象のテキストが空のため再生をスキップします。")
            return
        
        audio_content = self._synthesize_tts(text_to_synthesize, ai_id)
        if audio_content:
            print(f"[{ai_id}]: 音声を再生中...")
            self._play_audio(audio_content)
            print(f"[{ai_id}]: 再生終了。")

    def _synthesize_and_play_system_message(self, text_to_synthesize):
        """システムからのメッセージを音声合成し、再生する。標準的な声を使用。"""
        if not text_to_synthesize:
            print("[システム]: 合成対象のテキストが空のため再生をスキップします。")
            return

        # マークダウン記号を除去
        cleaned_text = self._clean_text_for_tts(text_to_synthesize)

        tts_client = texttospeech.TextToSpeechClient()
        p_audio = pyaudio.PyAudio()
        
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
            return
        
        stream = None
        try:
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True
            )
            print(f"[システム]: 音声を再生中...")
            stream.write(audio_content)
            print("再生終了。")
        except Exception as e:
            print(f"システムメッセージ再生エラー: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p_audio.terminate() # PyAudioインスタンスを終了

    def add_to_history(self, speaker, content): 
        """会話履歴に発言を追加する"""
        self.conversation_history.append({"speaker": speaker, "content": content})

    def calculate_facilitation_scores(self):
        """
        5つのファシリテーション手法のスコアを計算（各1点、計5点満点）
        LLMによる一括判定方式を使用
        
        Returns:
            tuple: (各項目のスコア辞書, 合計スコア)
        """
        scores = {
            "目的確認": 0,
            "役割分担": 0,
            "意見引き出し": 0,
            "議論整理": 0,
            "時間管理": 0
        }
        
        # ユーザーの発言のみを抽出（システムメッセージは除外）
        user_utterances = [
            msg['content'] for msg in self.conversation_history 
            if msg['speaker'] == self.username
        ]
        
        if not user_utterances:
            print("[採点]: ユーザーの発言がないため、スコアは0点です")
            return scores, 0
        
        print(f"[採点]: {len(user_utterances)}件の発言を分析中...")
        
        # LLMによる一括判定
        prompt = f"""以下のユーザーの発言リストから、5つのファシリテーション手法が実施されたかを判定してください。

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
以下の形式で、必ず数字（0または1）のみで回答してください。他の説明は不要です。

目的確認: [0または1]
役割分担: [0または1]
意見引き出し: [0または1]
議論整理: [0または1]
時間管理: [0または1]
"""
        
        try:
            # Gemini APIで判定
            response = self.gemini_model.generate_content([
                {"role": "user", "parts": [prompt]}
            ])
            
            if not response._result.candidates:
                print(f"[採点エラー]: Geminiからの応答がブロックされました")
                print(f"Safety feedback: {response.prompt_feedback}")
                # フォールバック: キーワード検出を使用
                return self._fallback_keyword_scoring(user_utterances)
            
            # レスポンスをパース
            response_text = response.text.strip()
            print(f"[採点]: LLM応答を受信: {response_text[:100]}...")
            
            # 各項目のスコアを抽出
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 数字のみを抽出
                    import re
                    match = re.search(r'[01]', value)
                    if match:
                        score_value = int(match.group())
                        if key in scores or key == "目的確認" or key == "目的の確認":
                            if "目的" in key:
                                scores["目的確認"] = score_value
                            elif "役割" in key:
                                scores["役割分担"] = score_value
                            elif "意見" in key:
                                scores["意見引き出し"] = score_value
                            elif "議論" in key or "整理" in key:
                                scores["議論整理"] = score_value
                            elif "時間" in key:
                                scores["時間管理"] = score_value
            
            total_score = sum(scores.values())
            print(f"[採点完了]: 合計 {total_score}/5点")
            for item, score in scores.items():
                print(f"  - {item}: {score}点")
            
            return scores, total_score
            
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
            tuple: (各項目のスコア辞書, 合計スコア)
        """
        print("[採点]: フォールバックモード（キーワード検出）を使用")
        
        scores = {
            "目的確認": 0,
            "役割分担": 0,
            "意見引き出し": 0,
            "議論整理": 0,
            "時間管理": 0
        }
        
        # 1. 目的確認
        purpose_keywords = ["ゴール", "目的", "議題", "決定", "話し合い", "議論", "テーマ"]
        purpose_phrases = ["について", "を決める", "を決定", "について話し合"]
        for utterance in user_utterances:
            if any(keyword in utterance for keyword in purpose_keywords) or \
               any(phrase in utterance for phrase in purpose_phrases):
                scores["目的確認"] = 1
                break
        
        # 2. 役割分担
        role_keywords = ["書記", "タイムキーパー", "役割", "分担", "記録"]
        for utterance in user_utterances:
            if any(keyword in utterance for keyword in role_keywords):
                scores["役割分担"] = 1
                break
        
        # 3. 意見引き出し
        elicitation_keywords = ["どう思いますか", "意見", "考え", "どうですか", "いかがですか", "どうでしょうか"]
        for utterance in user_utterances:
            if any(keyword in utterance for keyword in elicitation_keywords):
                scores["意見引き出し"] = 1
                break
        
        # 4. 議論整理
        summary_keywords = ["まとめ", "つまり", "要約", "整理", "まとめると", "まとめて"]
        for utterance in user_utterances:
            if any(keyword in utterance for keyword in summary_keywords):
                scores["議論整理"] = 1
                break
        
        # 5. 時間管理
        time_keywords = ["時間", "分", "残り", "あと", "残り時間", "時間が", "時間を"]
        time_phrases = ["そろそろ", "時間がない", "時間配分"]
        for utterance in user_utterances:
            if any(keyword in utterance for keyword in time_keywords) or \
               any(phrase in utterance for phrase in time_phrases):
                scores["時間管理"] = 1
                break
        
        total_score = sum(scores.values())
        print(f"[採点完了]: 合計 {total_score}/5点（フォールバック）")
        return scores, total_score

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
        
        # 採点を実行（0/1スコア）
        scores, total_score = self.calculate_facilitation_scores()
        report["ファシリテーション手法スコア"] = scores
        report["合計スコア"] = f"{total_score}/5点"
        
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
        
        print("\n--- GD簡易フィードバックレポート ---")
        for key, value in report.items():
            print(f"{key}: {value}")
        print("\nレポート生成が完了しました。")
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
                f.write(f"**ユーザー名**: {fullname}\n")
                f.write(f"**実施日時**: {timestamp.replace('_', ' ').replace('-', '/')}\n")
                f.write(f"**GDテーマ**: {self.gd_theme.splitlines()[0]}\n\n") # タイトルのみ保存
                
                # スコアを表示（最初に表示）
                if "ファシリテーション手法スコア" in report:
                    f.write("## ファシリテーション手法スコア\n\n")
                    scores = report["ファシリテーション手法スコア"]
                    for item, score in scores.items():
                        f.write(f"- **{item}**: {score}点\n")
                    f.write(f"\n**合計スコア**: {report.get('合計スコア', 'N/A')}\n\n")
                    f.write("---\n\n")
                
                for key, value in report.items():
                    # スコアは既に表示済みなのでスキップ
                    if key in ["ファシリテーション手法スコア", "合計スコア"]:
                        continue
                    f.write(f"## {key}\n\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"- **{k}**: {v}\n")
                    else:
                        f.write(f"{value}\n")
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
        
        # CSVファイルパスを返す（統制群の場合はCSVのみ、実験群の場合はMarkdownファイルパスを返す）
        return csv_filepath if csv_filepath else (filepath if experiment_group == "experimental" else None)

    def __del__(self):
        """GDManagerが終了する際にPyAudioリソースを解放する"""
        print("GDManagerの終了処理を実行中...")
        print("GDManagerが終了しました。")

    def _analyze_user_intent(self, user_text):
        """
        LLMを使い、ユーザーのテキストからファシリテーションの意図を判断する。
        """
        if self.roles_assigned:
            intent_list = ["一般的な発言", "特定の参加者への質問", "時間管理", "意見引き出し", "議題設定", "要約"]
        else:
            intent_list = ["一般的な発言", "特定の参加者への質問", "時間管理", "意見引き出し", "議題設定", "要約", "役割分担"]

        # LLMに意図の判断を依頼するプロンプトを構築
        messages = [
            {"role": "user", "parts": [
                f"以下の発言は、どのファシリテーションの意図に最も近いですか？\n"
                f"選択肢: {', '.join(intent_list)}\n"
                f"発言: {user_text}\n"
                f"最も近い意図を、選択肢の中から一つだけ返してください。もしどの意図にも当てはまらない場合は、「一般的な発言」と判断してください。選択肢以外の回答はしないでください。"
            ]}
        ]

        try:
            response = self.gemini_model.generate_content(messages)
            if response._result.candidates:
                # LLMの応答から、意図のテキストを取得
                return response.text.strip()
        except Exception as e:
            print(f"意図分析エラー: {e}")

        return "判断不能" # エラーや無効な応答の場合

    def _get_target_ai_from_text(self, text):
        """
        ユーザーのテキストから、特定のAI参加者を特定する。
        複数の呼び方（AI参加者A、Aさん、AI-Aなど）に対応する。
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
    
    def _parse_role_assignment(self, user_text):
        """
        LLMを使ってユーザーの発言から役割分担の指示を解析する
        
        Returns:
            dict: {ai_name: role} の辞書。ユーザーが具体的に指定した場合のみ値が入る。
                  指定がない場合は空の辞書を返す。
        """
        # AI参加者の名前リストを作成
        ai_names = [ai_id for ai_id in self.participants.keys() if ai_id != self.username]
        
        # LLMに役割分担の指示を解析させる
        prompt = f"""以下の発言から、役割分担の指示があるかどうかを判断してください。

【発言】
{user_text}

【利用可能な役割】
- 書記: 議事録を記録する役割
- タイムキーパー: 時間を管理する役割

【参加者名】
{', '.join(ai_names)}

【指示】
1. 発言に役割分担の指示が含まれているか判断してください
2. 具体的に「（参加者名）は（役割）をお願いします」のような指定がある場合、以下のJSON形式で返してください：
{{"参加者名": "役割名"}}

例1: 「田中さんは書記をお願いします」→ {{"田中": "書記"}}
例2: 「鈴木さんにタイムキーパーを任せます」→ {{"鈴木": "タイムキーパー"}}
例3: 「役割を決めましょう」→ {{}}

役割分担の指示がない場合、または具体的な指定がない場合は空のJSON {{}} を返してください。
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
                
                # JSON部分を抽出
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    json_str = json_match.group()
                    role_assignments = json.loads(json_str)
                    # 空の辞書でない場合のみ返す
                    if role_assignments:
                        # 参加者名が実際に存在するか確認
                        valid_assignments = {}
                        for ai_name, role in role_assignments.items():
                            if ai_name in ai_names and role in ["書記", "タイムキーパー"]:
                                valid_assignments[ai_name] = role
                        return valid_assignments
        except Exception as e:
            print(f"役割分担解析エラー: {e}")
        
        return {}

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
            time.sleep(0.3)  # 次の発言者への切り替え時間
        
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
        try:
            # GD開始後の最初の発言（ターンカウントが0）の場合のみ、タイムアウトを適用しない
            current_timeout = None if self.turn_count == 0 else 8
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
                    if not response.results or not response.results[0].alternatives:
                        continue
                    result = response.results[0]
                    if result.is_final:
                        user_text = result.alternatives[0].transcript.strip()
                        print(f"\n[あなた]: {user_text}")
                        break
        except Exception as e:
            print(f"音声認識エラーが発生しました: {e}")
            return False
        finally:
            p_audio.terminate() # 処理終了後に必ず解放

        # ユーザーの発言がなかった場合の処理
        if not user_text or user_text is None:
            # GD開始後の最初の発言で無発話だった場合、システムが何もせずに待機する
            if self.turn_count == 0:
                print("[システム]: ユーザーからの発言がありませんでした。お待ちしています。")
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
            
            task_for_ai = "会話が少し途切れているようです。会話履歴と文脈を理解して、ペルソナに忠実に自然に発言してください。議論を進めるための提案や、テーマに関する意見を述べても構いません。"
            # AI応答のループ（ストリーミング版を使用）
            for ai_name in ai_participants_to_respond:
                # AI発言開始を通知
                if speaker_changed_signal:
                    speaker_changed_signal.emit(ai_name)
                self.current_speaker = ai_name
                llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task_for_ai)
                self.add_to_history(ai_name, llm_response_text)
                time.sleep(0.3)
                
                # AIが発言した後、他のAIが反応する（自然な会話の流れを維持）
                if not self._process_ai_response_to_speech(ai_name, llm_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                    return False  # 時間切れの場合は終了
            return True

        # ユーザーの発言があった場合の処理
        self.add_to_history(self.username, user_text)
        self.turn_count += 1
        self.current_speaker = self.username
        
        # 役割分担の処理（ユーザーが具体的に指定した場合、または役割分担がまだ行われていない場合）
        if not self.roles_assigned:
            role_assignments = self._parse_role_assignment(user_text)
            
            # ユーザーが具体的に役割を指定した場合
            if role_assignments:
                for ai_name, role in role_assignments.items():
                    self.participants[ai_name]["assigned_role"] = role
                    if gd_thread is not None and hasattr(gd_thread, "role_updated"):
                        gd_thread.role_updated.emit(ai_name, role)
                self.roles_assigned = True
                # 役割が割り当てられたAIに発言させる
                last_ai_name = None
                last_response_text = None
                for ai_name, role in role_assignments.items():
                    task = f"ファシリテーターが「{ai_name}さんは{role}をお願いします」と指定しました。あなたは{role}という役割を担うことを確認してください。発言は簡潔にしてください。"
                    if speaker_changed_signal:
                        speaker_changed_signal.emit(ai_name)
                    self.current_speaker = ai_name
                    llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task)
                    self.add_to_history(ai_name, llm_response_text)
                    last_ai_name = ai_name
                    last_response_text = llm_response_text
                    time.sleep(0.3)
                
                # 最後のAIの発言に対して他のAIが反応する
                if last_ai_name and last_response_text:
                    if not self._process_ai_response_to_speech(last_ai_name, last_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                        return False  # 時間切れの場合は終了
                return True
            
            # 役割分担の意図があるかLLMで判断
            user_intent = self._analyze_user_intent(user_text)
            if user_intent == "役割分担":
                # 自動的に書記とタイムキーパーを割り当て
                all_ai_participants = [ai for ai in self.participants.keys() if ai != self.username]
                role_candidates = ["タイムキーパー", "書記"]
                random.shuffle(role_candidates)
                
                last_ai_name = None
                last_response_text = None
                for i, ai_name in enumerate(all_ai_participants):
                    if i < len(role_candidates):
                        assigned_role = role_candidates[i]
                        self.participants[ai_name]["assigned_role"] = assigned_role
                        if gd_thread is not None and hasattr(gd_thread, "role_updated"):
                            gd_thread.role_updated.emit(ai_name, assigned_role)
                        task = f"ファシリテーターが役割分担を促しました。あなたは「{assigned_role}」という役割を担うことを自己提案してください。発言は簡潔にしてください。"
                        if speaker_changed_signal:
                            speaker_changed_signal.emit(ai_name)
                        self.current_speaker = ai_name
                        llm_response_text = self._synthesize_and_play_ai_response_streaming(ai_name, task)
                        self.add_to_history(ai_name, llm_response_text)
                        last_ai_name = ai_name
                        last_response_text = llm_response_text
                        time.sleep(0.3)
                self.roles_assigned = True
                
                # 最後のAIの発言に対して他のAIが反応する
                if last_ai_name and last_response_text:
                    if not self._process_ai_response_to_speech(last_ai_name, last_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                        return False  # 時間切れの場合は終了
                return True

        ai_participants_to_respond = []
        task_for_ai = ""

        if self.turn_count == 1:
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
        # 自己紹介時（turn_count==1）はLLM応答生成を並列化して高速化
        if self.turn_count == 1 and len(ai_participants_to_respond) > 2:
            print(f"[システム]: {len(ai_participants_to_respond)}名のAI自己紹介を並列生成中...")
            
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
                time.sleep(0.3)
            
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
                kickoff_message = f"今回のファシリテーターは{self.username}さんです。テーマについて議論を始めてください。制限時間は{time_display}です。"
                self.conversation_history.append({"speaker": "システム", "content": kickoff_message})
                print(f"[システム]: {kickoff_message}")
                self._synthesize_and_play_system_message(kickoff_message)
                self.kickoff_announced = True
            
            # 自己紹介が終わったのでタイマーを開始
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
                time.sleep(0.3)  # 次の発言者への切り替え時間を短縮
                
                # AIが発言した後、他のAIが反応する（自己紹介や役割分担の場合はスキップ）
                # 自己紹介（turn_count==1）や役割分担（roles_assigned==False）以外の場合のみ、他のAIが反応する
                if self.turn_count > 1:
                    if not self._process_ai_response_to_speech(ai_name, llm_response_text, speaker_changed_signal, gd_thread, is_chain_reaction=False):
                        return False  # 時間切れの場合は終了

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

【論点・課題】❓
• （未解決の問題や議論が必要なポイント、なければ「なし」）

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

    # GUIアプリケーションのインスタンスを作成
    app = QApplication(sys.argv)

    # GUIウィンドウを作成
    gui_window = GDReportWindow()
    
    # 開発モード: 自動的にデフォルト名字でスタート（実験群/統制群選択は手動）
    # 開発モードでは自動開始を無効化（実験群/統制群選択が必要なため）
    # if DEV_MODE:
    #     print(f"[開発モード] デフォルト名字「{DEV_DEFAULT_USERNAME}」で自動開始します")
    #     # 少し遅延させてGUI初期化を待つ
    #     from PySide6.QtCore import QTimer
    #     QTimer.singleShot(500, lambda: gui_window.user_input_screen.username_input.setText(DEV_DEFAULT_USERNAME))
    #     QTimer.singleShot(600, lambda: gui_window.user_input_screen._on_start_clicked())
    
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
            # テーマタイトルを画面・議事録エリアに表示
            gui_window.set_theme(theme)
            gui_window.gd_screen.set_theme(theme)
            # ユーザーの役職をGUIに反映
            gui_window.gd_screen.update_participant_role(username, "ファシリテーター")
            
            # まずローディングを消し、GD画面を完全に表示してからシステム発話を行う
            gui_window.gd_screen.hide_loading()
            QApplication.processEvents()  # UI更新を確実に実行
            
            # GD開始時の初期メッセージ（自己紹介の案内など）を再生
            # ※この処理中はTTSでメインスレッドがブロックされるが、すでにローディングは消えている
            context.manager._initialize_gd()
            
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