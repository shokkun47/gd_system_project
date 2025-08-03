import os
import time
import pyaudio
import wave
import numpy as np # 音声データ処理用（tts_test.pyで使いましたが、無音追加しないなら不要。ただしnumpyは他の場所で使う可能性あり残します）

# --- dotenv のインポートと読み込み ---
from dotenv import load_dotenv # 追加: .envファイルから環境変数を読み込むため
load_dotenv() # 追加: これで.envファイルから環境変数が読み込まれる

# Google Cloud Speech-to-Text (ASR) APIクライアント
from google.cloud import speech_v1p1beta1 as speech 

# Google Cloud Text-to-Speech (TTS) APIクライアント
from google.cloud import texttospeech_v1beta1 as texttospeech 

# OpenAI API (LLM) クライアント
import openai 

# --- 設定 ---
# プロジェクトルートのパスを動的に取得
# voice_chat_mvp.py は src/tests/ にあるため、3階層上に遡る
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 音声I/O（Input/Output）に関する共通設定
RATE = 16000     # サンプリングレート (Hz)
CHUNK = 1024     # 音声データを処理する際のバッファサイズ
FORMAT = pyaudio.paInt16 # 音声データのフォーマット
CHAS = 1     # 音声チャンネル数

# 生成・保存するファイル名 (パスを修正)
USER_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "user_input_mvp.wav")
AI_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "ai_output_mvp.wav")

# 言語設定
LANGUAGE_CODE_ASR = "ja-JP"
LANGUAGE_CODE_TTS = "ja-JP"
AI_VOICE_NAME = "ja-JP-Wavenet-C" # AIの声
LLM_MODEL = "gpt-3.5-turbo"     # LLMモデル

# --- APIクライアントの初期化 ---
try:
    speech_client = speech.SpeechClient()         # Google Speech-to-Text用クライアント
    tts_client = texttospeech.TextToSpeechClient() # Google Text-to-Speech用クライアント
    openai_client = openai.OpenAI()               # OpenAI API用クライアント
    print("全APIクライアントを初期化しました。")
except Exception as e:
    print(f"APIクライアントの初期化に失敗しました。環境変数またはAPIキーを確認してください: {e}")
    exit()

# --- 音声入力 (ASR) 関連関数 ---
def record_user_input(seconds=5):
    """
    ユーザーのマイクから指定秒数音声を録音し、バイトデータとして返します。
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
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
    p.terminate()

    audio_content = b''.join(frames)
    
    # WAVファイルとして保存（デバッグ用）
    try:
        os.makedirs(os.path.dirname(USER_AUDIO_FILE), exist_ok=True) # フォルダが存在しない場合作成
        wf = wave.open(USER_AUDIO_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_content)
        wf.close()
    except Exception as e:
        print(f"音声ファイルの保存中にエラーが発生しました: {e}")
    
    return audio_content

def transcribe_audio(audio_content):
    """
    録音された音声データをGoogle Speech-to-Text APIでテキストに変換します。
    """
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=LANGUAGE_CODE_ASR,
        enable_automatic_punctuation=True
    )
    
    try:
        response = speech_client.recognize(config=config, audio=audio)
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"ASRエラー: {e}")
        return ""

# --- LLMによる応答生成 ---
def get_llm_response(user_text):
    """
    ユーザーのテキストをLLMに送信し、AIの応答テキストを生成します。
    """
    messages = [
        {"role": "system", "content": "あなたはユーザーの質問に簡潔に答えるAIアシスタントです。"},
        {"role": "user", "content": user_text}
    ]
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=100, # 応答の長さを短めに設定
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLMエラー: {e}")
        return "すみません、今は応答できません。"

# --- 音声出力 (TTS) 関連関数 ---
def synthesize_and_play(text_to_synthesize):
    """
    テキストをGoogle Text-to-Speech APIで音声に合成し、再生します。
    """
    synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
    voice = texttospeech.VoiceSelectionParams(
        language_code=LANGUAGE_CODE_TTS,
        name=AI_VOICE_NAME,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_content = response.audio_content # 生成された生音声データ
        
    except Exception as e:
        print(f"TTS合成エラー: {e}")
        return

    # --- 音声をPyAudioで再生 ---
    p = pyaudio.PyAudio()
    stream = None
    try:
        # WAVファイルとして一時保存（再生のために必要）
        os.makedirs(os.path.dirname(AI_AUDIO_FILE), exist_ok=True) # フォルダが存在しない場合作成
        wf = wave.open(AI_AUDIO_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_content) # WAVファイルに直接書き込み
        wf.close()
        
        # 保存したWAVファイルを読み込み、再生ストリームを開く
        wf_read = wave.open(AI_AUDIO_FILE, 'rb')
        stream = p.open(format=p.get_format_from_width(wf_read.getsampwidth()),
                        channels=wf_read.getnchannels(),
                        rate=wf_read.getframerate(),
                        output=True)
        
        # 音声データをCHUNKサイズずつ読み込み、ストリームに書き込んで再生
        data = wf_read.readframes(CHUNK)
        print("音声を再生中...")
        while data:
            stream.write(data)
            data = wf_read.readframes(CHUNK)
        
        print("再生終了。")
            
    except Exception as e:
        print(f"TTS再生エラー: {e}")
        print("ヒント: スピーカーが接続されているか、音量が適切か確認してください。")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

# --- メインループ ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("OPENAI_API_KEY"):
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY) が設定されていません。")
        exit()

    print("--- 1対1 AI音声チャット MVP開始 ---")
    print("「終了」と話すとチャットが終了します。")

    while True:
        recorded_audio = record_user_input(seconds=5)
        user_text = transcribe_audio(recorded_audio)
        
        print(f"あなた: {user_text}")

        if "終了" in user_text:
            print("チャットを終了します。")
            break

        if user_text:
            ai_response_text = get_llm_response(user_text)
            print(f"AI: {ai_response_text}")

            if ai_response_text:
                synthesize_and_play(ai_response_text)
        else:
            print("何も認識されませんでした。もう一度お話しください。")
        
        time.sleep(0.5)