import os    # OS機能（環境変数の読み込みなど）
import time    # 時間に関する処理（待機など）
import pyaudio # マイクからの音声入力とスピーカーからの音声再生
import wave    # WAVファイルの読み書き

# Google Cloud Speech-to-Text (ASR) APIクライアント
from google.cloud import speech_v1p1beta1 as speech 

# Google Cloud Text-to-Speech (TTS) APIクライアント
from google.cloud import texttospeech_v1beta1 as texttospeech 

# OpenAI API (LLM) クライアント
import openai 

# --- 設定 ---
# 音声I/O（Input/Output）に関する共通設定
RATE = 16000     # サンプリングレート (Hz): 1秒間に音声を何回デジタル化するか。ASR/TTSの推奨値。
CHUNK = 1024     # 音声データを処理する際のバッファサイズ（一度に読み込むフレーム数）。
FORMAT = pyaudio.paInt16 # 音声データのフォーマット。16ビット整数（一般的なオーディオ形式）。
CHANNELS = 1     # 音声チャンネル数。1はモノラル、2はステレオ。マイク録音は通常モノラル。

# 生成・保存するファイル名
USER_AUDIO_FILE = "audio_dada/user_input_mvp.wav" # ユーザーの録音音声を一時保存するファイル名（デバッグ用）
AI_AUDIO_FILE = "audio_data/ai_output_mvp.wav"   # AIの生成音声を一時保存するファイル名（デバッグ用）

# 言語設定
LANGUAGE_CODE_ASR = "ja-JP" # ASR（音声認識）で認識する言語コード（日本語）
LANGUAGE_CODE_TTS = "ja-JP" # TTS（音声合成）で生成する言語コード（日本語）
AI_VOICE_NAME = "ja-JP-Wavenet-C" # AIの声。Google TTSのボイス名（ja-JP-Wavenet-A, B, C...など）。
LLM_MODEL = "gpt-3.5-turbo"     # LLMとして使用するモデル名。gpt-4oは高性能だが高コスト。

# --- APIクライアントの初期化 ---
# ここで各クラウドサービスのAPIクライアント（通信するための窓口）を作成します。
# これらは、環境変数（GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY）を使って自動認証されます。
try:
    speech_client = speech.SpeechClient()         # Google Speech-to-Text用クライアント
    tts_client = texttospeech.TextToSpeechClient() # Google Text-to-Speech用クライアント
    openai_client = openai.OpenAI()               # OpenAI API用クライアント（ChatGPTなど）
    print("全APIクライアントを初期化しました。")
except Exception as e:
    # どのクライアントかの初期化に失敗した場合、エラーメッセージを表示してプログラムを終了。
    print(f"APIクライアントの初期化に失敗しました。環境変数またはAPIキーを確認してください: {e}")
    exit() # プログラムを終了する

# --- 音声入力 (ASR) 関連関数 ---
def record_user_input(seconds=5):
    """
    ユーザーのマイクから指定秒数音声を録音し、バイトデータとして返します。
    """
    p = pyaudio.PyAudio() # PyAudioオブジェクトを初期化（オーディオデバイスへのアクセス開始）
    # マイク入力ストリームを開く
    stream = p.open(format=FORMAT,     # 音声フォーマット（paInt16 = 16ビット整数）
                    channels=CHANNELS, # チャンネル数（モノラル）
                    rate=RATE,         # サンプリングレート
                    input=True,        # 入力（マイク）ストリームとして開く
                    frames_per_buffer=CHUNK) # バッファサイズ

    print(f"ユーザー発言を録音中 ({seconds}秒)...")
    frames = [] # 録音された音声データを格納するリスト

    # 指定秒数分の音声データをCHUNKごとに読み込み、リストに追加
    for _ in range(0, int(RATE / CHUNK * seconds)):
        try:
            data = stream.read(CHUNK) # マイクからCHUNK分のデータを読み込む
            frames.append(data)       # 読み込んだデータをリストに追加
        except IOError as e:
            # 録音中にエラーが発生した場合（マイクの問題など）
            print(f"録音中にエラーが発生しました: {e}")
            break # ループを中断

    # ストリームを停止・閉じる
    stream.stop_stream() # 録音ストリームを停止
    stream.close()       # ストリームを閉じる
    p.terminate()        # PyAudioのリソースを解放

    audio_content = b''.join(frames) # 録音された全データをバイト列に結合

    # WAVファイルとして保存（デバッグ用）。録音内容を後で確認できるようにする。
    wf = wave.open(USER_AUDIO_FILE, 'wb') # WAVファイルを開く（書き込みモード）
    wf.setnchannels(CHANNELS)           # チャンネル数を設定
    wf.setsampwidth(p.get_sample_size(FORMAT)) # サンプル幅（バイト数）を設定
    wf.setframerate(RATE)               # フレームレートを設定
    wf.writeframes(audio_content)       # 録音されたバイトデータをファイルに書き込む
    wf.close()                          # ファイルを閉じる
    
    return audio_content # 録音されたバイトデータを返す

def transcribe_audio(audio_content):
    """
    録音された音声データをGoogle Speech-to-Text APIでテキストに変換します。
    """
    # 認識対象の音声データを設定
    audio = speech.RecognitionAudio(content=audio_content)
    # 音声認識の設定
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # 音声データのエンコーディング形式
        sample_rate_hertz=RATE,                                  # サンプリングレート
        language_code=LANGUAGE_CODE_ASR,                         # 言語コード
        enable_automatic_punctuation=True                        # 自動で句読点付与（読みやすくなる）
    )
    
    try:
        # Speech-to-Text APIに認識リクエストを送信
        response = speech_client.recognize(config=config, audio=audio)
        if response.results: # 認識結果が存在する場合
            # 最も確信度の高い結果のテキスト部分を取り出す
            return response.results[0].alternatives[0].transcript
        return "" # 結果がなければ空文字列を返す
    except Exception as e:
        # API呼び出し中にエラーが発生した場合
        print(f"ASRエラー: {e}")
        return "" # エラー時は空文字列を返す

# --- LLMによる応答生成関連関数 ---
def get_llm_response(user_text):
    """
    ユーザーのテキストをLLMに送信し、AIの応答テキストを生成します。
    """
    # LLMへの入力は「メッセージのリスト」形式。AIの役割設定とユーザーの質問を含む。
    messages = [
        {"role": "system", "content": "あなたはユーザーの質問に簡潔に答えるAIアシスタントです。"},
        {"role": "user", "content": user_text} # ユーザーの最新の発言
    ]
    try:
        # OpenAI APIのchat.completions.createを呼び出し、LLMから応答を生成
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,    # 使用するモデル
            messages=messages,  # メッセージ履歴
            temperature=0.7,    # 応答のランダム性（0.0で予測可能、1.0で創造的）
            max_tokens=100,     # 生成する応答の最大トークン数
        )
        # 生成された応答テキストを取得し、前後の空白を削除して返す
        return response.choices[0].message.content.strip()
    except Exception as e:
        # LLM呼び出し中のエラー（APIキー問題、接続問題、レート制限など）
        print(f"LLMエラー: {e}")
        return "すみません、今は応答できません。" # エラー時のフォールバック応答

# --- 音声出力 (TTS) 関連関数 ---
def synthesize_and_play(text_to_synthesize):
    """
    テキストをGoogle Text-to-Speech APIで音声に合成し、再生します。
    """
    # 音声合成の入力データを作成
    synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)
    # 声の選択設定
    voice = texttospeech.VoiceSelectionParams(
        language_code=LANGUAGE_CODE_TTS,
        name=AI_VOICE_NAME,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # VOICE_NAMEに合わせて調整（例:FEMALE, MALE）
    )
    # 音声形式の設定
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, # LINEAR16形式（PyAudioで再生しやすい）
        sample_rate_hertz=RATE                              # サンプリングレート
    )

    try:
        # TTS APIを呼び出し、音声合成を実行
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_content = response.audio_content # 合成された音声データ（バイト列）
        
    except Exception as e:
        print(f"TTS合成エラー: {e}")
        return

    # --- 音声をPyAudioで再生 ---
    p = pyaudio.PyAudio() # PyAudioオブジェクトを初期化
    stream = None         # ストリーム変数を初期化
    try:
        # WAVファイルとして一時保存（再生のために必要）
        wf = wave.open(AI_AUDIO_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_content)
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
        # ストリームとPyAudioオブジェクトを適切に終了させる（エラー時でも実行される）
        if stream: # streamがNoneでなければ
            stream.stop_stream()
            stream.close()
        p.terminate() # PyAudioのリソースを解放

# --- メインループ ---
if __name__ == "__main__":
    # プログラムが直接実行された場合に、APIキーの環境変数をチェックし、メインのチャットループを開始します。
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("OPENAI_API_KEY"):
        print("エラー: 必要な環境変数 (GOOGLE_APPLICATION_CREDENTIALS, OPENAI_API_KEY) が設定されていません。")
        exit() # 環境変数がなければ終了

    print("--- 1対1 AI音声チャット MVP開始 ---")
    print("「終了」と話すとチャットが終了します。")

    while True: # 無限ループで対話を続ける
        # 1. ユーザーの発話を録音し、テキストに変換する
        recorded_audio = record_user_input(seconds=5) # 5秒間録音
        user_text = transcribe_audio(recorded_audio)   # 録音データをテキストに認識
        
        print(f"あなた: {user_text}") # ユーザーの認識テキストを表示

        # ユーザーが「終了」と話したらループを抜ける
        if "終了" in user_text:
            print("チャットを終了します。")
            break

        if user_text: # ユーザーが何かテキストとして認識された場合のみ処理
            # 2. LLMにユーザーの発話に対する応答を生成してもらう
            ai_response_text = get_llm_response(user_text)
            print(f"AI: {ai_response_text}") # AIの応答テキストを表示

            # 3. AIの応答を音声合成し、再生する
            if ai_response_text: # AIの応答テキストが空でなければ
                synthesize_and_play(ai_response_text)
        else:
            print("何も認識されませんでした。もう一度お話しください。") # 音声が認識されない場合
        
        time.sleep(0.5) # 次のループ（録音開始）までの短い待機時間