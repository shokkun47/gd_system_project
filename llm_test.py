import os
import openai # pip install openai

# --- 設定 ---
# 環境変数 OPENAI_API_KEY が設定されている必要があります
try:
    # openai.OpenAI() クライアントを初期化
    # API_KEYは環境変数 OPENAI_API_KEY から自動的に読み込まれます
    client = openai.OpenAI() 
    print("OpenAI API クライアントを初期化しました。")
except Exception as e:
    print(f"OpenAI APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
    print("エラーメッセージ: {e}")
    print("ヒント: 環境変数 OPENAI_API_KEY が正しく設定されていますか？")
    exit()

# --- LLMとの対話を行う関数 ---
def get_llm_response(prompt_text, model="gpt-3.5-turbo"): # 最初はコストが低いgpt-3.5-turboを推奨
    messages = [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": prompt_text}
    ]

    print(f"\nLLMにリクエストを送信中... (モデル: {model})")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # 応答のランダム性 (0.0: 決定論的, 1.0: 創造的)
            max_tokens=150,  # 生成する応答の最大トークン数
        )
        llm_response = response.choices[0].message.content.strip()
        print("\n--- LLMからの応答 ---")
        print(f"応答テキスト: {llm_response}")
        return llm_response
    except openai.APIConnectionError as e:
        print(f"OpenAI APIへの接続に失敗しました: {e}")
        print("ヒント: インターネット接続を確認してください。")
    except openai.RateLimitError as e:
        print(f"APIレート制限に達しました: {e}")
        print("ヒント: しばらく待ってから再試行するか、APIキーの利用状況を確認してください。")
    except openai.AuthenticationError as e:
        print(f"OpenAI API認証に失敗しました: {e}")
        print("ヒント: 環境変数 OPENAI_API_KEY が正しく設定されているか確認してください。")
    except Exception as e:
        print(f"LLMの呼び出し中に予期せぬエラーが発生しました: {e}")
    return None

# --- スクリプトの実行 ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。OpenAI APIキーを設定してください。")
    else:
        # 好きな質問をしてみましょう
        user_question = input("LLMに質問を入力してください: ")
        get_llm_response(user_question)
        print("\nLLMテストスクリプト完了。")