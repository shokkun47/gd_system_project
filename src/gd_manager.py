import os
import time
import json
# import openai  # <-- OpenAIはもう使いません

import sys 

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
import wave    # WAVファイルの読み書き（一時ファイルの保存用）
import numpy as np # 音声データ処理（無音生成、形式変換など）

# --- システム共通設定 ---
# gd_manager.py は src/ ディレクトリにあるため、プロジェクトルートパスを正確に取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 音声I/Oに関する共通設定
RATE = 16000     # サンプリングレート (Hz) - Google ASR/TTSの推奨値
CHUNK = 1024     # 音声データを処理する際のバッファサイズ（フレーム数）。PyAudioで一度に読み書きする単位。
FORMAT = pyaudio.paInt16 # 音声データのフォーマット。16ビット整数。
CHANNELS = 1     # 音声チャンネル数。1はモノラル。

# 一時的な音声ファイルを保存するパス (audio_dataフォルダはプロジェクトルート直下)
USER_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "user_input.wav")
AI_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "ai_output.wav")

# LLMとTTSの言語・モデル設定
LANGUAGE_CODE_ASR = "ja-JP"     # ASR（音声認識）で認識する言語
LANGUAGE_CODE_TTS = "ja-JP"     # TTS（音声合成）で生成する言語
DEFAULT_AI_VOICE_NAME = "ja-JP-Wavenet-C" # AIの声のデフォルト設定（Google TTSのボイス名）
# LLM_MODEL を Gemini のモデル名に変更
GEMINI_MODEL = "gemini-1.5-flash-latest" # または "gemini-pro" など。高速版を推奨。

class GDManager:
    """
    グループディスカッションの進行を管理する中央オーケストレータークラス。
    Gemini APIを利用してユーザーとのGDシミュレーションを管理する。
    """
    def __init__(self, gd_theme="新製品のマーケティング戦略", num_ai_participants=3):
        print("GDManagerを初期化中...")
        
        self.gd_theme = gd_theme
        self.num_ai_participants = num_ai_participants
        self.conversation_history = [] 
        self.current_phase = "導入" 
        self.time_limit_minutes = 20 
        self.start_time = time.time() 
        self.turn_count = 0 
        self.current_speaker = "システム" 

        # --- APIクライアントの初期化 ---
        try:
            self.speech_client = speech.SpeechClient()         # ASRクライアント
            self.tts_client = texttospeech.TextToSpeechClient() # TTSクライアント
            
            # --- Gemini APIの認証とクライアント初期化 ---
            # .envファイルから GOOGLE_API_KEY 環境変数を読み込みます
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # <-- GOOGLE_API_KEY を .env に設定すること
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL) # Geminiモデルをロード
            print("全APIクライアントを GDManager 内で初期化しました。")
        except Exception as e:
            print(f"エラー: APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
            print("ヒント: GOOGLE_APPLICATION_CREDENTIALS と GOOGLE_API_KEY が正しく設定されていますか？")
            exit()
        
        self.p_audio = pyaudio.PyAudio() 

        # --- デバッグ情報 (動作確認後に削除してOK) ---
        print(f"DEBUG: sys.path = {sys.path}") 
        # Geminiではopenai.__file__やdir(self.openai_client)は不要

        # --- 参加者とペルソナの設定 ---
        self.participants = {
            "ユーザー": {"role": "ユーザー", "persona": "あなたはGDを円滑に進行し、結論に導く責任があるファシリテーターです。"}
        }
        self.ai_voice_map = {} 
        for i in range(num_ai_participants):
            ai_id = f"AI参加者{chr(ord('A') + i)}" 
            persona_text = self._get_default_ai_persona(ai_id)
            self.participants[ai_id] = {"role": ai_id, "persona": persona_text}
            
            # AI参加者ごとに異なる声色を割り当てる (Google TTSのボイス名)
            # ja-JP-Wavenet-A, B, C, D, E, F などから選択
            if i == 0: self.ai_voice_map[ai_id] = "ja-JP-Wavenet-A" # 女性の声
            elif i == 1: self.ai_voice_map[ai_id] = "ja-JP-Wavenet-B" # 男性の声
            elif i == 2: self.ai_voice_map[ai_id] = "ja-JP-Wavenet-D" # 別の女性の声
            else: self.ai_voice_map[ai_id] = DEFAULT_AI_VOICE_NAME

        print("GDManagerの初期化が完了しました。")
        self._initialize_gd() 

    def _get_default_ai_persona(self, ai_id):
        """AI参加者ごとのデフォルトペルソナを設定する"""
        if ai_id == "AI参加者A":
            return "あなたはGDの参加者であるAI参加者Aです。積極的に意見を出し、具体的な提案を重視します。常に新しい視点を提供します。"
        elif ai_id == "AI参加者B":
            return "あなたはGDの参加者であるAI参加者Bです。常に批判的な視点から問題点を指摘し、議論を深掘りします。リスク評価や実現可能性の検討が得意です。"
        elif ai_id == "AI参加者C":
            return "あなたはGDの参加者であるAI参加者Cです。協調的で、議論の要約や確認を好み、参加者間の合意形成を促します。円滑なコミュニケーションを重視します。"
        else:
            return "あなたはGDの参加者であるAIです。議論に積極的に貢献します。"

    def _initialize_gd(self):
        """GD開始時の初期メッセージを発言させる"""
        initial_message = f"GDを始めます。本日のテーマは「{self.gd_theme}」です。皆様、よろしくお願いいたします。"
        self.conversation_history.append({"speaker": "システム", "content": initial_message})
        print(f"[システム]: {initial_message}")
        self._synthesize_and_play_system_message(initial_message) 

        first_ai_id = "AI参加者A"
        ai_response_text = self._get_ai_response(first_ai_id, "GD開始の挨拶と、簡単な自己紹介、今日の議論への意気込みを簡潔に述べてください。", include_current_history=False)
        self.conversation_history.append({"speaker": first_ai_id, "content": ai_response_text})
        print(f"[{first_ai_id}]: {ai_response_text}")
        self._synthesize_and_play_ai_response(ai_response_text, first_ai_id) 
        self.current_speaker = first_ai_id 

    def _generate_ai_prompt(self, ai_id, current_task_for_ai, include_current_history=True):
        """
        特定のAI参加者に対するLLMへのプロンプト（指示文）を構築する。
        Gemini APIの形式に合わせる。
        """
        # Geminiはロールを'user'/'model'で区別することが多いため、調整
        messages_content = []
        
        # システム指示
        system_instruction_content = (
            f"あなたはGDの参加者である{ai_id}です。あなたのペルソナは以下の通りです:\n"
            f"{self.participants[ai_id]['persona']}\n"
            "これまでのGD履歴と、あなたへの指示を踏まえ、適切にGDに参加する発言を生成してください。\n"
            "あなたの発言は、ペルソナに忠実であり、議論の目的達成に貢献するものでなければなりません。\n"
            "簡潔かつ自然な日本語で発言し、不要な説明やAIとしての言及は避けてください。"
        )
        # Geminiはsystem roleを直接サポートしていないため、user roleに含める
        messages_content.append({"role": "user", "parts": [system_instruction_content]})
        messages_content.append({"role": "model", "parts": ["了解しました。"]}) # AIの初期応答をシミュレート
        
        # 過去の会話履歴
        if include_current_history:
            # Geminiは会話のロールを'user'と'model'で交互にする必要があるので、調整
            for msg in self.conversation_history:
                if msg['speaker'] == "システム": continue # システムメッセージはLLMに渡さない
                # Geminiでは'user'と'model'が交互にくる必要がある
                role_gemini = "user" if msg['speaker'] == "ユーザー" else "model"
                messages_content.append({"role": role_gemini, "parts": [msg['content']]})

        # 最新のタスク指示
        messages_content.append({"role": "user", "parts": [
            f"現在のGDフェーズ: {self.current_phase}\n"
            f"残り時間: {int((self.time_limit_minutes * 60 - (time.time() - self.start_time)) / 60)}分\n"
            f"あなたのタスク: {current_task_for_ai}"
        ]})
        
        return messages_content

    def _get_ai_response(self, ai_id, task_for_ai, include_current_history=True):
        """
        特定のAI参加者（ai_id）の発言をLLM（Gemini）に生成させる。
        """
        messages = self._generate_ai_prompt(ai_id, task_for_ai, include_current_history)
        
        print(f" (GDマネージャー -> Geminiへの指示 for {ai_id}): {task_for_ai[:50]}...") 
        
        try:
            # Gemini API呼び出し
            response = self.gemini_model.generate_content(messages)
            
            # 応答がブロックされた場合のエラー処理
            if not response._result.candidates:
                print(f"エラー: Geminiからの応答がブロックされました for {ai_id}. Safety feedback: {response.prompt_feedback}")
                return f"（{ai_id}）すみません、不適切な内容と判断されたため発言できません。"

            return response.text.strip()
        except Exception as e:
            print(f"エラー: Geminiの呼び出し中に予期せぬエラーが発生しました: {e}")
            return f"（{ai_id}）予期せぬエラーのため発言できません。"

    def _record_user_input(self, seconds=5):
        """ユーザーのマイクから指定秒数音声を録音し、バイトデータとして返す。"""
        stream = self.p_audio.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
        print(f"ユーザー発言を録音中 ({seconds}秒)...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * seconds)):
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except IOError as e:
                print(f"録音中にエラーが発生しました: {e}")
                break
        stream.stop_stream()
        stream.close()
        return b''.join(frames)

    def _transcribe_audio(self, audio_content):
        """録音された音声データをGoogle Speech-to-Text APIでテキストに変換。"""
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=LANGUAGE_CODE_ASR,
            enable_automatic_punctuation=True
        )
        try:
            response = self.speech_client.recognize(config=config, audio=audio) 
            if response.results:
                return response.results[0].alternatives[0].transcript
            return ""
        except Exception as e:
            print(f"ASRエラー: {e}")
            return ""

    def _synthesize_and_play_ai_response(self, text_to_synthesize, ai_id):
        """AIの応答テキストを音声合成し、再生する。"""
        voice_name = self.ai_voice_map.get(ai_id, DEFAULT_AI_VOICE_NAME)

        synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_TTS,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL 
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE
        )

        try:
            response = self.tts_client.synthesize_speech( 
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content 
            
        except Exception as e:
            print(f"TTS合成エラー: {e}")
            return

        # 音声をPyAudioで再生
        stream = None 
        try:
            os.makedirs(os.path.dirname(AI_AUDIO_FILE), exist_ok=True) 
            wf = wave.open(AI_AUDIO_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p_audio.get_sample_size(FORMAT)) 
            wf.setframerate(RATE)
            wf.writeframes(audio_content) 
            wf.close()
            
            stream = self.p_audio.open(format=FORMAT, 
                                      channels=CHANNELS,
                                      rate=RATE,
                                      output=True)
            print("音声を再生中...")
            stream.write(audio_content) 
            print("再生終了。")
                
        except Exception as e:
            print(f"TTS再生エラー: {e}")
        finally:
            if stream: 
                stream.stop_stream()
                stream.close()

    def _synthesize_and_play_system_message(self, text_to_synthesize):
        """システムからのメッセージを音声合成し、再生する。AIの声を借りる。"""
        voice_name = DEFAULT_AI_VOICE_NAME 
        synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_TTS,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE
        )
        try:
            response = self.tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            audio_content = response.audio_content
            stream = self.p_audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
            stream.write(audio_content)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"システムメッセージ再生エラー: {e}")


    def process_user_input(self): 
        """
        ユーザーからの音声を録音・認識し、GDの進行を管理する。
        AIからの応答をトリガーし、音声を再生する。
        """
        recorded_audio = self._record_user_input(seconds=5)
        user_text = self._transcribe_audio(recorded_audio)
        
        if not user_text:
            print("[システム]: 何も認識されませんでした。もう一度お話しください。")
            self._synthesize_and_play_system_message("何も認識されませんでした。もう一度お話しください。") 
            return True # GDを続行

        self.add_to_history("ユーザー", user_text) 
        self.turn_count += 1
        self.current_speaker = "ユーザー" 

        print(f"\n[ユーザー]: {user_text}")

        # --- GDマネージャー（中央オーケストレーター）の主な判断ロジック ---
        ai_participants_to_respond = []
        for ai_name in ["AI参加者A", "AI参加者B", "AI参加者C"]: # AIのIDを正確に指定
            ai_participants_to_respond.append(ai_name)
        
        for ai_name in ai_participants_to_respond:
            task_for_ai = "直前の議論（ユーザーの発言も含む）を踏まえ、自身の役割とペルソナに沿って発言してください。"
            
            llm_response_text = self._get_ai_response(ai_name, task_for_ai) 
            
            self.add_to_history(ai_name, llm_response_text) # 履歴に追加
            
            print(f"[{ai_name}]: {llm_response_text}")
            self._synthesize_and_play_ai_response(llm_response_text, ai_name)
            
            time.sleep(0.5) # AIの発言間隔をシミュレート (会話の自然さのため)
            self.current_speaker = ai_name # 発言者を更新

        # GDの終了条件判定など
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False # GDを終了する
        return True # GDを続行する

    def add_to_history(self, speaker, content): 
        """会話履歴に発言を追加する"""
        self.conversation_history.append({"speaker": speaker, "content": content})
        print(f"履歴に追加: [{speaker}] {content[:30]}...") 

    def generate_simple_feedback_report(self):
        """
        GD終了後に簡易フィードバックレポートを生成する。
        """
        print("\n--- 簡易フィードバックレポート生成中 ---")
        report = {}
        
        # 1. 総発言数と時間
        total_utterances = len(self.conversation_history)
        duration_minutes = int((time.time() - self.start_time) / 60)
        report["総GD時間(分)"] = duration_minutes
        report["総発言数"] = total_utterances

        # 2. 参加者ごとの発言回数
        speaker_counts = {}
        for entry in self.conversation_history:
            speaker_counts[entry['speaker']] = speaker_counts.get(entry['speaker'], 0) + 1
        report["参加者別発言回数"] = speaker_counts

        # 3. GDフェーズの最終到達点（このMVPではまだ詳細管理がないが、将来の機能）
        report["最終到達フェーズ"] = self.current_phase

        # 4. LLMによる簡単な振り返り（全会話履歴を元に分析）
        feedback_prompt = (
            "以下のGDログについて、ファシリテーター（ユーザー）の役割を中心に、議論の進行状況や雰囲気に関する簡単な振り返り（50文字程度）と、改善点に関する短い示唆（50文字程度）を生成してください。\n\n"
            + "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in self.conversation_history])
        )
        try:
            # Gemini APIでフィードバック生成
            feedback_messages = [
                {"role": "user", "parts": ["あなたはGDのパフォーマンスを評価し、簡潔な振り返りと示唆を提供するアシスタントです。"]},
                {"role": "model", "parts": ["承知しました。"]},
                {"role": "user", "parts": [feedback_prompt]}
            ]
            llm_feedback_response = self.gemini_model.generate_content(feedback_messages)
            
            if not llm_feedback_response._result.candidates:
                llm_feedback = f"Geminiからのフィードバック応答がブロックされました。Safety feedback: {llm_feedback_response.prompt_feedback}"
            else:
                llm_feedback = llm_feedback_response.text.strip()
            
            report["LLMからの総括と示唆"] = llm_feedback
        except Exception as e:
            report["LLMからの総括と示唆"] = f"フィードバック生成中にエラー: {e}"

        print("\n--- GD簡易フィードバックレポート ---")
        for key, value in report.items():
            print(f"{key}: {value}")
        print("\nレポート生成が完了しました。")
        return report

    # GDマネージャーのリソースを解放するメソッド
    def __del__(self):
        """GDManagerが終了する際にPyAudioリソースを解放する"""
        print("GDManagerの終了処理を実行中...")
        if self.p_audio:
            self.p_audio.terminate() # PyAudioリソースを解放
        print("GDManagerが終了しました。")


# --- スクリプトの実行（GDManagerの動作確認用） ---
if __name__ == "__main__":
    print("--- GDManagerの統合テスト開始 ---")
    
    # 環境変数が設定されているか最終チェック (dotenvが読み込む)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_API_KEY"): # GOOGLE_API_KEYもチェック
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY) が設定されていません。")
        exit()

    manager = GDManager() # GDManagerを初期化
    
    print("\n--- 複数AI参加者とのGDシミュレーション開始 ---")
    print("あなたがファシリテーターです。マイクに向かって話してください（「終了」と話すと終了）。")

    running_gd = True
    while running_gd:
        try:
            # process_user_inputを引数なしで呼び出し、ユーザーの音声入力を処理させる
            running_gd = manager.process_user_input() 
            
            # GD終了条件のチェックはprocess_user_input内で行われる (時間制限)
            # もしユーザーが「終了」と話した場合はprocess_user_input内で終了処理が行われる
            if not running_gd: 
                break

        except KeyboardInterrupt: # Ctrl+Cで中断された場合
            print("\nGDが中断されました。")
            running_gd = False
            
        if not running_gd: # ループを抜けるために明示的に
            break

    # GD終了後にフィードバックレポートを生成
    final_report = manager.generate_simple_feedback_report()
    print("\nGDシミュレーションが完了しました。")