import os
import pyaudio
import wave
import threading
import queue
import time

# --- 変更点1: enums のインポートを削除 ---
from google.cloud import speech_v1p1beta1 as speech
# from google.cloud.speech_v1p1beta1 import enums  <-- この行を削除

# --- 設定 ---
AUDIO_FILE_NAME = "user_input.wav" # 音声を一時的に保存するファイル名 (デバッグ用)
RATE = 16000                     # サンプリングレート (Hz) - Google Speech-to-Textの推奨
CHUNK = 1024                     # 1度に読み込むフレーム数
RECORD_SECONDS = 5               # 録音する秒数（例として5秒）
LANGUAGE_CODE = "ja-JP"          # 認識する言語

# --- ASRクライアントの初期化 ---
try:
    speech_client = speech.SpeechClient()
    print("Google Speech-to-Text クライアントを初期化しました。")
except Exception as e:
    print(f"ASRクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
    print(f"エラーメッセージ: {e}")
    print("ヒント: GCPの認証情報JSONファイルのパスが、環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていますか？")
    exit()

# --- 音声録音とASR処理の関数 ---
def record_and_recognize():
    audio = pyaudio.PyAudio()
    
    # マイク入力ストリームを開く
    try:
        stream = audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"マイク入力ストリームの開始に失敗しました。マイクが接続されているか、権限が許可されているか確認してください: {e}")
        return

    print(f"{RECORD_SECONDS}秒間録音を開始します。マイクに向かって話してください...")
    frames = []

    # 録音
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except IOError as e:
            print(f"録音中にエラーが発生しました（IOError: {e}）。マイクが適切に機能しているか確認してください。")
            break

    print("録音終了。認識中...")

    # ストリームを停止・閉じる
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 音声をWAVファイルとして保存 (デバッグ用。後で削除してもOK)
    try:
        wf = wave.open(AUDIO_FILE_NAME, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        print(f"音声ファイルの保存中にエラーが発生しました: {e}")


    # --- 音声認識の実行 ---
    # 保存したファイルを読み込む
    try:
        with open(AUDIO_FILE_NAME, 'rb') as audio_file:
            content = audio_file.read()
    except FileNotFoundError:
        print("エラー: 録音ファイルが見つかりません。録音が正しく行われなかった可能性があります。")
        return

    # 音声認識リクエストの作成
    audio_config = speech.RecognitionConfig(
        # --- 変更点2: enums を使わない形に変更 ---
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # speechオブジェクトから直接参照
        # または encoding='LINEAR16', # 文字列で直接指定
        sample_rate_hertz=RATE,                                  # サンプリングレート
        language_code=LANGUAGE_CODE,                             # 言語
        enable_automatic_punctuation=True                        # 自動句読点付与（オプション）
    )
    
    audio_content = speech.RecognitionAudio(content=content)

    # 同期認識
    try:
        response = speech_client.recognize(config=audio_config, audio=audio_content)

        if response.results:
            # 最も可能性の高い結果を表示
            print("\n--- 認識結果 ---")
            for result in response.results:
                print(f"認識テキスト: {result.alternatives[0].transcript}")
        else:
            print("\n認識結果がありませんでした。マイクの音声が小さかったか、ノイズが多かった可能性があります。")
    except Exception as e:
        print(f"\n音声認識APIの呼び出し中にエラーが発生しました: {e}")
        print("認証情報、APIの有効化、またはインターネット接続を確認してください。")

# --- スクリプトの実行 ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("エラー: 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。GCPの認証情報を設定してください。")
    else:
        record_and_recognize()