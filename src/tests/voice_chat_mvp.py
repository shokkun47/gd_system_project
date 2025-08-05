import os
import time
import pyaudio
import wave
import numpy as np 

# dotenvのインポートと読み込み
from dotenv import load_dotenv 
load_dotenv()

# Google Cloud Speech-to-Text (ASR)
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud Text-to-Speech (TTS)
from google.cloud import texttospeech_v1beta1 as texttospeech

# Google Gemini APIクライアントをインポート
import google.generativeai as genai

# --- 設定 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

USER_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "user_input_mvp.wav")
AI_AUDIO_FILE = os.path.join(PROJECT_ROOT, "audio_data", "ai_output_mvp.wav")

LANGUAGE_CODE_ASR = "ja-JP"
LANGUAGE_CODE_TTS = "ja-JP"
AI_VOICE_NAME = "ja-JP-Wavenet-C"
GEMINI_MODEL = "gemini-1.5-flash-latest"

# --- APIクライアントの初期化 ---
try:
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    print("全APIクライアントを初期化しました。")
except Exception as e:
    print(f"APIクライアントの初期化に失敗しました。環境変数またはAPIキーを確認してください: {e}")
    exit()

# --- 音声入力 (ASR) ---
def record_user_input(seconds=5):
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
    
    try:
        os.makedirs(os.path.dirname(USER_AUDIO_FILE), exist_ok=True)
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

# --- LLMによる応答生成 (修正済み) ---
def get_llm_response(messages, model_name=GEMINI_MODEL):
    messages_copy = messages.copy()
    try:
        response = gemini_model.generate_content(messages_copy)
        
        if not response._result.candidates:
            print(f"LLMエラー: Geminiからの応答がブロックされました。Safety feedback: {response.prompt_feedback}")
            return "すみません、不適切な内容と判断されたため応答できません。"

        ai_response_text = response.text.strip()
        
        # messagesリストは呼び出し元で更新するため、ここでは返さない
        return ai_response_text
    except Exception as e:
        print(f"LLMエラー: Geminiの呼び出し中に予期せぬエラーが発生しました: {e}")
        return "すみません、今は応答できません。"

# --- 音声出力 (TTS) ---
def synthesize_and_play(text_to_synthesize):
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
        audio_content = response.audio_content
    except Exception as e:
        print(f"TTS合成エラー: {e}")
        return

    p = pyaudio.PyAudio()
    stream = None
    try:
        os.makedirs(os.path.dirname(AI_AUDIO_FILE), exist_ok=True)
        wf = wave.open(AI_AUDIO_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_content)
        wf.close()
        
        wf_read = wave.open(AI_AUDIO_FILE, 'rb')
        stream = p.open(format=p.get_format_from_width(wf_read.getsampwidth()),
                        channels=wf_read.getnchannels(),
                        rate=wf_read.getframerate(),
                        output=True)
        data = wf_read.readframes(CHUNK)
        print("音声を再生中...")
        while data:
            stream.write(data)
            data = wf_read.readframes(CHUNK)
        
        print("再生終了。")
    except Exception as e:
        print(f"TTS再生エラー: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

# --- メインループ ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_API_KEY"):
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY) が設定されていません。")
        exit()

    print("--- 1対1 AI音声チャット MVP開始 ---")
    print("「終了」と話すとチャットが終了します。")
    
    conversation_history = []
    
    while True:
        recorded_audio = record_user_input(seconds=5)
        user_text = transcribe_audio(recorded_audio)
        
        print(f"あなた: {user_text}")

        if "終了" in user_text:
            print("チャットを終了します。")
            break

        if user_text:
            # ユーザーの発言を履歴に追加
            conversation_history.append({"role": "user", "parts": [user_text]})
            
            # LLMに応答を生成してもらう (messagesリスト全体を渡す)
            ai_response_text = get_llm_response(conversation_history) 
            
            if ai_response_text:
                # AIの応答を履歴に追加
                conversation_history.append({"role": "model", "parts": [ai_response_text]})
                print(f"AI: {ai_response_text}")

                if ai_response_text:
                    synthesize_and_play(ai_response_text)
        else:
            print("何も認識されませんでした。")
        
        time.sleep(0.5)