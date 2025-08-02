import os
import time
import pyaudio
import wave

# Google Cloud Speech-to-Text (ASR)
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud Text-to-Speech (TTS)
from google.cloud import texttospeech_v1beta1 as texttospeech

# OpenAI API (LLM)
import openai

# --- 設定 ---
# 音声I/O設定
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ファイル名
USER_AUDIO_FILE = "user_input_mvp.wav"
AI_AUDIO_FILE = "ai_output_mvp.wav"

# 言語設定
LANGUAGE_CODE_ASR = "ja-JP"
LANGUAGE_CODE_TTS = "ja-JP"
AI_VOICE_NAME = "ja-JP-Wavenet-C" # AIの声 (男性の声の例)
LLM_MODEL = "gpt-3.5-turbo"     # LLMモデル (最初はコストの低いものを推奨)

# --- APIクライアントの初期化 ---
try:
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
    openai_client = openai.OpenAI()
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
    
    # WAVファイルとして保存（デバッグ用）
    wf = wave.open(USER_AUDIO_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_content)
    wf.close()
    
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

# --- LLMによる応答生成 ---
def get_llm_response(user_text):
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

# --- 音声出力 (TTS) ---
def synthesize_and_play(text_to_synthesize):
    synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
    voice = texttospeech.VoiceSelectionParams(
        language_code=LANGUAGE_CODE_TTS,
        name=AI_VOICE_NAME,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # VOICE_NAMEに合わせて調整
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

        # 音声をPyAudioで再生
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)
        
        # LINEAR16データをCHUNKサイズで再生
        # response.audio_content はバイナリデータなので、そのままstream.write()に渡せる
        chunk_size = CHUNK * 2 # 16bit = 2bytes, so CHUNK * 2 bytes
        
        for i in range(0, len(audio_content), chunk_size):
            stream.write(audio_content[i:i + chunk_size])
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"TTS再生エラー: {e}")


# --- メインループ ---
if __name__ == "__main__":
    # 環境変数が設定されているかチェック
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("OPENAI_API_KEY"):
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY) が設定されていません。")
        exit()

    print("--- 1対1 AI音声チャット MVP開始 ---")
    print("「終了」と話すと終了します。")

    while True:
        # ユーザーの発話録音とテキスト化
        recorded_audio = record_user_input(seconds=5)
        user_text = transcribe_audio(recorded_audio)
        
        print(f"あなた: {user_text}")

        if "終了" in user_text:
            print("チャットを終了します。")
            break

        if user_text: # ユーザーが何か話した場合
            # LLMに応答を生成してもらう
            ai_response_text = get_llm_response(user_text)
            print(f"AI: {ai_response_text}")

            # AIの応答を音声合成・再生
            if ai_response_text:
                synthesize_and_play(ai_response_text)
        else:
            print("何も認識されませんでした。")
        
        time.sleep(0.5) # 次のループまでの短い待機時間