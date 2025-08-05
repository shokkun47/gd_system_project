import os
# import openai  # OpenAIはもう使いません

# Google Gemini APIクライアントをインポート
import google.generativeai as genai 

# .envファイルから環境変数を読み込むため (ファイルの冒頭に置く)
from dotenv import load_dotenv 
load_dotenv()

# --- 設定 ---
# LLMモデルをGeminiのモデル名に変更
GEMINI_MODEL = "gemini-1.5-flash-latest" # または "gemini-1.5-flash-latest" など、より高速なモデルも選択可能

# --- APIクライアントの初期化 ---
try:
    # Gemini APIの認証設定
    # .envファイルから GOOGLE_API_KEY 環境変数を読み込みます
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
    
    # Geminiモデルをロード
    client = genai.GenerativeModel(GEMINI_MODEL)
    print(f"Gemini API クライアントを初期化しました (モデル: {GEMINI_MODEL})。")
except Exception as e:
    print(f"Gemini APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
    print(f"エラーメッセージ: {e}")
    print("ヒント: 環境変数 GOOGLE_API_KEY が正しく設定されていますか？")
    exit()

# --- LLMとの対話を行う関数 ---
def get_llm_response(messages, model_name=GEMINI_MODEL): # 引数名をmodel_nameに変更
    """
    メッセージのリストをGemini LLMに送信し、応答を返します。
    """
    print(f"\nLLMにリクエストを送信中... (モデル: {model_name})")
    try:
        # Gemini API呼び出し
        # messagesリストの形式がOpenAIとは異なります（後述の解説を参照）
        response = client.generate_content(messages)
        
        # 応答がブロックされた場合のエラー処理
        if not response._result.candidates:
            print(f"LLMエラー: Geminiからの応答がブロックされました。Safety feedback: {response.prompt_feedback}")
            return "すみません、不適切な内容と判断されたため応答できません。"

        llm_response = response.text.strip()
        print("\n--- LLMからの応答 ---")
        print(f"応答テキスト: {llm_response}")
        return llm_response
    except Exception as e:
        print(f"LLMエラー: Geminiの呼び出し中に予期せぬエラーが発生しました: {e}")
        return "すみません、今は応答できません。"

# --- スクリプトの実行 ---
if __name__ == "__main__":
    # GOOGLE_API_KEY が設定されているかチェック
    if not os.getenv("GOOGLE_API_KEY"):
        print("エラー: 環境変数 GOOGLE_API_KEY が設定されていません。Gemini APIキーを設定してください。")
        exit()

    print("--- Gemini LLM テスト開始 ---")
    print("「終了」と入力するとチャットが終了します。")

    # 会話履歴を保持するリスト
    # Geminiのメッセージ形式はOpenAIとは異なります
    # ロールは 'user' と 'model' を交互に繰り返す必要があります
    conversation_history = [] 
    
    # システムプロンプトやAIの初期設定は、最初の'user'メッセージに含めるか、
    # 各ターンで'user'メッセージに含める形で渡すのが一般的です。
    # ここではシンプルな対話のため、直接ユーザーの質問から始めます。

    while True:
        user_input = input("あなた: ")
        
        if user_input == "終了":
            print("チャットを終了します。")
            break

        if user_input:
            # ユーザーの発言を履歴に追加 (Gemini形式)
            conversation_history.append({"role": "user", "parts": [user_input]})
            
            # LLMに応答を生成してもらう
            # get_llm_response 関数はmessagesリスト全体を受け取る
            ai_response_text = get_llm_response(conversation_history) 
            
            if ai_response_text:
                # AIの応答を履歴に追加 (Gemini形式)
                conversation_history.append({"role": "model", "parts": [ai_response_text]})
                print(f"AI: {ai_response_text}")
            else:
                print("AIからの応答がありませんでした。")
        else:
            print("何も入力されませんでした。")