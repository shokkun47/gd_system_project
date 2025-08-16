import os
import time
import random

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

def get_random_gd_theme():
    """
    ランダムな課題解決型GDテーマを選択して返す。
    """
    themes = [
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
    
    return random.choice(themes)

class GDManager:
    """
    グループディスカッションの進行を管理する中央オーケストレータークラス。
    Gemini APIを利用してユーザーとのGDシミュレーションを管理する。
    """
    def __init__(self, num_ai_participants=3):
        print("GDManagerを初期化中...")
        
        self.gd_theme = get_random_gd_theme()
        self.num_ai_participants = num_ai_participants
        self.conversation_history = [] 
        self.current_phase = "導入" 
        self.time_limit_minutes = 2
        self.start_time = time.time() 
        self.turn_count = 0 
        self.current_speaker = "システム" 
        self.roles_assigned = False

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

        # --- 参加者とペルソナの設定 ---
        # 日本語の一般的な名前の候補リストを作成
        name_candidates = [
            "田中", "佐藤", "鈴木", "高橋", "渡辺", "伊藤", "山本", "中村", "小林", "加藤",
            "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
            "山崎", "森", "池田", "橋本", "阿部", "石川", "中島", "小野", "藤井", "原田",
            "岡田", "後藤", "長谷川", "村上", "近藤", "前田", "石田", "坂本", "遠藤", "青木"
        ]        
        # 候補からランダムに3名を選出
        selected_ai_names = random.sample(name_candidates, self.num_ai_participants)
        
        self.participants = {
            "ユーザー": {"role": "ユーザー", "persona": "あなたはGDを円滑に進行し、結論に導く責任があるファシリテーターです。"}
        }
        self.ai_voice_map = {} 

        for i, ai_id in enumerate(selected_ai_names):
            # ランダムに選ばれた名前に対応するペルソナを割り当てる
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
        """AI参加者ごとのデフォルトペルソナをランダムに設定する"""
        persona_options = [
            f"あなたはGDの参加者である{ai_id}です。積極的に意見を出し、具体的な提案を重視します。常に新しい視点を提供します。",
            f"あなたはGDの参加者である{ai_id}です。常に批判的な視点から問題点を指摘し、議論を深掘りします。リスク評価や実現可能性の検討が得意です。",
            f"あなたはGDの参加者である{ai_id}です。協調的で、議論の要約や確認を好み、参加者間の合意形成を促します。円滑なコミュニケーションを重視します。",
            f"あなたはGDの参加者である{ai_id}です。議論に積極的に貢献します。"
        ]
        # 毎回ランダムなペルソナを割り当てる
        return random.choice(persona_options)

    def _initialize_gd(self):
        """GD開始時の初期メッセージを発言させる"""
        initial_message = f"GDを始めます。本日のテーマは「{self.gd_theme}」です。皆様、まずは簡単な自己紹介と、今日の議論への意気込みを簡潔に述べていただけますでしょうか？"
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
        system_instruction_content = (
            f"あなたはGDの参加者である{ai_id}です。あなたのペルソナは以下の通りです:\n"
            f"{self.participants[ai_id]['persona']}\n" 
            "これまでのGD履歴と、あなたへの指示を踏まえ、適切にGDに参加する発言を生成してください。\n"
            "あなたの発言は、ペルソナに忠実であり、議論の目的達成に貢献するものでなければなりません。\n"
            "簡潔かつ自然な日本語で発言し、不要な説明やAIとしての言及は避けてください。"
        )
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
        
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False

        recorded_audio = self._record_user_input(seconds=5)
        user_text = self._transcribe_audio(recorded_audio)
        
        if not user_text:
            system_message = "ファシリテーターからの発言が無いようです。GDの進行を促してください。"
            print(f"[システム]: {system_message}")
            self._synthesize_and_play_system_message(system_message)
            return True

        self.add_to_history("ユーザー", user_text) 
        self.turn_count += 1
        self.current_speaker = "ユーザー" 

        print(f"\n[ユーザー]: {user_text}")

        # --- GDマネージャー（中央オーケストレーター）の主な判断ロジック ---
        # --- ステップ1で追加したメソッドを呼び出し、ユーザーの意図を分析 ---
        user_intent = self._analyze_user_intent(user_text)
        print(f"(デバッグ用) ユーザーのファシリテーション意図: {user_intent}")
        self._update_gd_phase(user_intent)

        # --- ここからが新しいAIターン交代ロジック ---
        ai_participants_to_respond = []

        if self.turn_count == 1: # GD開始後の最初のユーザー発言（自己紹介）
            # 参加者リストからAIのみを抽出
            all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]

            for ai_name in all_ai_participants:
                task_for_ai = "ファシリテーターが自己紹介を終えました。あなたもペルソナに沿って一言で自己紹介をしてください。"
            
                llm_response_text = self._get_ai_response(ai_name, task_for_ai)
                self.add_to_history(ai_name, llm_response_text)
                print(f"[{ai_name}]: {llm_response_text}")
                self._synthesize_and_play_ai_response(llm_response_text, ai_name)
                time.sleep(0.5)
                self.current_speaker = ai_name
            return True

        elif "特定の参加者への質問" in user_intent:
            # ユーザーが特定のAIに質問した場合、そのAIだけを応答させる
            target_ai_id = self._get_target_ai_from_text(user_text) # ユーザーの発言からAI名を特定する関数
            print(target_ai_id)
            if target_ai_id:
                ai_participants_to_respond.append(target_ai_id)
                task_for_ai = f"ファシリテーターからの質問に答えてください。発言は簡潔に答えてください。"

        elif "役割分担" in user_intent:
            # 役割の候補リストと定義
            role_candidates = ["タイムキーパー", "書記", "アイデアマン"]
            random.shuffle(role_candidates) # リストをシャッフル
            all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]

            for i, ai_name in enumerate(all_ai_participants):
                if i < len(role_candidates):
                    assigned_role = role_candidates[i]
                    self.participants[ai_name]["assigned_role"] = assigned_role
                    llm_response_text = self._get_ai_response(ai_name, f"ファシリテーターが役割分担を促しました。あなたは「{assigned_role}」という役割を担うことを自己提案してください。発言は簡潔にしてください。")
                    self.add_to_history(ai_name, llm_response_text)
                    print(f"[{ai_name}: {llm_response_text}]")
                    self._synthesize_and_play_ai_response(llm_response_text, ai_name)
                    time.sleep(0.5)
            
            self.roles_assigned = True
            return True

        elif "時間管理" in user_intent:
            timekeeper = None
            for ai_name in self.participants:
                if ai_name in self.participants:
                    if ai_name != "ユーザー" and self.participants[ai_name].get("assigned_role") == "タイムキーパー":
                        timekeeper = ai_name
                        break

            if timekeeper:
                ai_participants_to_respond.append(timekeeper)
                task_for_ai = f"ファシリテーターが時間管理を促しました。現在の残り時間を報告し、議論の進捗について簡潔にコメントしてください。"
            else:
                all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]
                ai_participants_to_respond = random.sample(all_ai_participants, 1)
                task_for_ai = f"ファシリテーターが時間管理を促しました。現在の残り時間を報告し、議論の進捗について簡潔にコメントしてください。"

        elif "意見引き出し" in user_intent or "議題設定" in user_intent:
            all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]
            ai_participants_to_respond = random.sample(all_ai_participants, 2)
            task_for_ai = "直前の議論（ユーザーの発言も含む）を踏まえ、ペルソナに沿って発言してください。発言は簡潔にしてください。" 
        
        elif "要約" in user_intent:
            collaborator = None
            for ai_name in self.participants:
                if ai_name != "ユーザー" and self.participants[ai_name].get("assigned_role") == "書記":
                    collaborator = ai_name
                    break
            
            if collaborator:
                ai_participants_to_respond.append(collaborator)
                task_for_ai = f"ファシリテーターが要約を促しました。これまでの議論の要点をまとめてください。発言は簡潔にしてください。"

            else:
                all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]
                ai_participants_to_respond = random.sample(all_ai_participants, 1)
                task_for_ai = f"ファシリテーターが要約を促しました。これまでの議論の要点をまとめてください。発言は簡潔にしてください。"

        else: # それ以外の発言（一般的な発言やGDの停滞など）
            all_ai_participants = [ai for ai in self.participants.keys() if ai != "ユーザー"]
            num_respondents = random.randint(1, len(all_ai_participants)) # 1人から3人までランダムに選ぶ
            ai_participants_to_respond = random.sample(all_ai_participants, num_respondents)
            
            for ai_name in ai_participants_to_respond:
                if time.time() - self.start_time > self.time_limit_minutes * 60:
                    print("\n--- GD終了: 制限時間になりました ---")
                    return False 

                task_for_ai = "直前の議論を踏まえ、ペルソナに沿って自由に発言してください。発言は簡潔にしてください。"         
                llm_response_text = self._get_ai_response(ai_name, task_for_ai)
                self.add_to_history(ai_name, llm_response_text)
                print(f"[{ai_name}]: {llm_response_text}")
                self._synthesize_and_play_ai_response(llm_response_text, ai_name)
                time.sleep(0.5)
                self.current_speaker = ai_name
            
            return True

        # AIからの応答を順次生成・処理
        for ai_name in ai_participants_to_respond:

            if time.time() - self.start_time > self.time_limit_minutes * 60:
                print("\n--- GD終了: 制限時間になりました ---")
                return False

            llm_response_text = self._get_ai_response(ai_name, task_for_ai)
            self.add_to_history(ai_name, llm_response_text)
            print(f"[{ai_name}]: {llm_response_text}")
            self._synthesize_and_play_ai_response(llm_response_text, ai_name)
            time.sleep(0.5) # AIの発言間隔をシミュレート (会話の自然さのため)
            self.current_speaker = ai_name # 発言者を更新

        # GDの終了条件判定など
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False 
        
        return True # GDを続行する

    def add_to_history(self, speaker, content): 
        """会話履歴に発言を追加する"""
        self.conversation_history.append({"speaker": speaker, "content": content})

    def generate_simple_feedback_report(self):
        """
        GD終了後に簡易フィードバックレポートを生成する。
        """
        print("\n--- 簡易フィードバックレポート生成中 ---")
        report = {}
        
        total_utterances = len(self.conversation_history) - 1
        duration_minutes = int((time.time() - self.start_time) / 60)
        report["総GD時間(分)"] = duration_minutes
        report["総発言数"] = total_utterances

        speaker_counts = {}
        for entry in self.conversation_history:
            if entry["speaker"] != "システム":
                speaker_counts[entry['speaker']] = speaker_counts.get(entry['speaker'], 0) + 1
        report["参加者別発言回数"] = speaker_counts

        report["最終到達フェーズ"] = self.current_phase

        feedback_prompt = (
            "以下のGDログについて、ファシリテーター（ユーザー）の役割を中心に、議論の進行状況や雰囲気に関する簡単な振り返り（50文字程度）と、改善点に関する短い示唆（50文字程度）を生成してください。\n\n"
            + "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in self.conversation_history])
        )
        try:
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

    def __del__(self):
        """GDManagerが終了する際にPyAudioリソースを解放する"""
        print("GDManagerの終了処理を実行中...")
        if self.p_audio:
            self.p_audio.terminate() # PyAudioリソースを解放
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
            if ai_id != "ユーザー":
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

    def _update_gd_phase(self, user_intent):
        """ ユーザーの意図に基づいてGDフェーズを更新する """
        if self.current_phase == "導入" and user_intent == "意見引き出し":
            self.current_phase = "意見発散"
            print(f"\n--- フェーズ移行: {self.current_phase} ---")
        elif self.current_phase == "意見発散" and user_intent == "要約":
            self.current_phase = "意見収束"
            print(f"\n--- フェーズ移行: {self.current_phase} ---")
        elif self.current_phase == "意見収束" and user_intent == "要約":
            self.current_phase = "決定・結論"
            print(f"\n--- フェーズ移行: {self.current_phase} ---")

# --- スクリプトの実行（GDManagerの動作確認用） ---
if __name__ == "__main__":
    print("--- GDManagerの統合テスト開始 ---")
    
    # 環境変数が設定されているか最終チェック (dotenvが読み込む)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_API_KEY"): # GOOGLE_API_KEYもチェック
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY) が設定されていません。")
        exit()

    manager = GDManager() # GDManagerを初期化
    
    print("\n--- 複数AI参加者とのGDシミュレーション開始 ---")
    print("あなたがファシリテーターです。マイクに向かって話してください。")

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

    print("\nGDシミュレーションが完了しました。")
    manager.generate_simple_feedback_report()