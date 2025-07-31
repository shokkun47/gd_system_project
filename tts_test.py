import os
import time
import pyaudio # pyaudioをインポート
import wave    # waveモジュールをインポート

from google.cloud import texttospeech_v1beta1 as texttospeech # Cloud TTSのクライアントライブラリ

# --- 設定 ---
TEXT_TO_SAY = "こんにちは。これはGoogle音声合成のテストです。うまく聞こえますか？"
VOICE_NAME = "ja-JP-Wavenet-A" # 使用する声。例: ja-JP-Wavenet-A, ja-JP-Wavenet-B など
LANGUAGE_CODE = "ja-JP"      # 言語コード
OUTPUT_FILENAME = "tts_output.wav" # 生成された音声を保存するファイル名

# --- PyAudioの設定 (asr_test.py と合わせる) ---
RATE = 16000 # サンプリングレート
CHUNK = 1024 # バッファサイズ

# --- TTSクライアントの初期化 ---
try:
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Text-to-Speech クライアントを初期化しました。")
except Exception as e:
    print(f"TTSクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
    print(f"エラーメッセージ: {e}")
    print("ヒント: GCPの認証情報JSONファイルのパスが、環境変数 GOOGLE_APPLICATION_CREDENTIALS が正しく設定されていますか？")
    exit()

# --- 音声合成と再生の関数 ---
def synthesize_and_play(text_to_synthesize, voice_name, lang_code, output_file):
    synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)

    # 声の設定
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE # FEMALE, MALE, NEUTRAL (ボイス名に合わせる)
    )

    # 音声の設定
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, # LINEAR16形式 (再生しやすい)
        sample_rate_hertz=RATE # サンプリングレート
    )

    print(f"「{text_to_synthesize}」を合成中...")
    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
    except Exception as e:
        print(f"音声合成APIの呼び出し中にエラーが発生しました: {e}")
        print("ヒント: 認証情報、APIの有効化、インターネット接続、または指定したVOICE_NAMEが有効か確認してください。")
        return

    # 音声をWAVファイルとして保存
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1) # モノラル
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16)) # 16-bit
            wf.setframerate(audio_config.sample_rate_hertz)
            wf.writeframes(response.audio_content)
        print(f"音声を '{output_file}' に保存しました。")
    except Exception as e:
        print(f"音声ファイルの保存中にエラーが発生しました: {e}")

    # 音声をPyAudioで再生
    p = pyaudio.PyAudio()
    stream = None
    try:
        # WAVファイルとして読み込み、ストリームを開く
        with wave.open(output_file, 'rb') as wf:
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(CHUNK)
            print("音声を再生中...")
            while data:
                stream.write(data)
                data = wf.readframes(CHUNK)

            print("再生終了。")
    except Exception as e:
        print(f"音声の再生中にエラーが発生しました: {e}")
        print("ヒント: スピーカーが接続されているか、音量が適切か確認してください。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- スクリプトの実行 ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("エラー: 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。GCPの認証情報を設定してください。")
    else:
        # VOICE_NAME を確認
        # Google Cloud Text-to-Speechの利用可能なボイスリストを参考に、好みのボイス名に変更できます
        # 例: ja-JP-Wavenet-A (女性), ja-JP-Wavenet-B (男性), ja-JP-Wavenet-C (男性), ja-JP-Wavenet-D (女性), ja-JP-Wavenet-E (女性), ja-JP-Wavenet-F (男性)
        # VOICE_NAME = "ja-JP-Wavenet-C" # 必要に応じて変更
        synthesize_and_play(TEXT_TO_SAY, VOICE_NAME, LANGUAGE_CODE, OUTPUT_FILENAME)
        time.sleep(1) # 終了前の少しの待機
        print("\nTTSテストスクリプト完了。")
