import os
import time
import json # 参加者のペルソナや設定を扱う場合に利用
import openai # OpenAI APIを扱うためのライブラリ

# .envファイルから環境変数を読み込むため
# プロジェクトルートに.envファイルを置いて、そこにAPIキーなどを設定します
from dotenv import load_dotenv

# --- 初期設定 (.envファイルから環境変数を読み込む) ---
load_dotenv() 

# --- MockOpenAIClient は不要になりますので、このコードでは使用しません ---

class GDManager:
    """
    グループディスカッションの進行を管理する中央オーケストレータークラス。
    AI参加者の発言を調整し、GDの流れを制御します。
    """
    def __init__(self, gd_theme="新製品のマーケティング戦略", num_ai_participants=3):
        print("GDManagerを初期化中...")
        
        # --- GDの基本設定 ---
        self.gd_theme = gd_theme # GDのテーマ
        self.num_ai_participants = num_ai_participants # AI参加者の数
        
        # 会話履歴を保持するリスト
        # 各要素は {"role": "speaker_id", "content": "発言内容"} の形式を想定
        self.conversation_history = [] 
        
        # GDの現在のフェーズ（例: "導入", "意見発散", "意見収束", "決定"）
        self.current_phase = "導入" 
        
        # 参加者（AIとユーザー）のペルソナや役割を定義
        self.participants = {
            "ユーザー": {"role": "ユーザー", "persona": "あなたはファシリテーターです。GDを円滑に進行し、結論に導く責任があります。"},
        }
        for i in range(num_ai_participants):
            ai_id = f"AI参加者{chr(ord('A') + i)}" # AI参加者A, AI参加者B, ...
            self.participants[ai_id] = {"role": ai_id, "persona": self._get_default_ai_persona(ai_id)}
        
        # GDの制限時間と開始時刻
        self.time_limit_minutes = 20
        self.start_time = time.time()
        
        # 現在の発言者（初期はシステムまたはユーザーから）
        self.current_speaker = "システム" 

        # --- OpenAI API クライアントの初期化 ---
        try:
            # 環境変数 OPENAI_API_KEY からAPIキーを自動的に読み込む
            self.ai_client = openai.OpenAI() 
            print("OpenAI API クライアントを GDManager 内で初期化しました。")
        except Exception as e:
            print(f"エラー: OpenAI APIクライアントの初期化に失敗しました。環境変数を確認してください: {e}")
            print("ヒント: 環境変数 OPENAI_API_KEY が正しく設定されているか確認してください。")
            exit() # 初期化失敗時はプログラムを終了

        print("GDManagerの初期化が完了しました。")
        self._initialize_gd() # GD開始時の初期処理

    def _get_default_ai_persona(self, ai_id):
        """AI参加者ごとのデフォルトペルソナを設定する"""
        if ai_id == "AI参加者A":
            return "あなたはGDの参加者であるAI参加者Aです。積極的に意見を出し、具体的な提案を重視します。過去の成功事例に詳しいです。"
        elif ai_id == "AI参加者B":
            return "あなたはGDの参加者であるAI参加者Bです。常に批判的な視点から問題点を指摘し、議論を深掘りします。リスク評価が得意です。"
        elif ai_id == "AI参加者C":
            return "あなたはGDの参加者であるAI参加者Cです。協調的で、議論の要約や確認を好み、合意形成を促します。参加者間の意見調整を試みます。"
        else:
            return "あなたはGDの参加者であるAIです。議論に積極的に貢献します。"

    def _initialize_gd(self):
        """GD開始時の初期メッセージを発言させる"""
        initial_message = f"GDを始めます。本日のテーマは「{self.gd_theme}」です。皆様、よろしくお願いいたします。"
        self.conversation_history.append({"speaker": "システム", "content": initial_message})
        print(f"[システム]: {initial_message}")
        time.sleep(1) # 再生時間シミュレート

        # AI参加者に初期の自己紹介や意気込みを発言させる（オプション）
        # ここでは、AI参加者Aに最初に話させます。
        first_ai_id = "AI参加者A"
        ai_response_text = self._get_ai_response(first_ai_id, "GD開始の挨拶と、簡単な自己紹介、今日の議論への意気込みを簡潔に述べてください。", include_current_history=False)
        self.conversation_history.append({"speaker": first_ai_id, "content": ai_response_text})
        print(f"[{first_ai_id}]: {ai_response_text}")
        time.sleep(1)
        self.current_speaker = first_ai_id # 最初のAIが話したので、次はユーザーに期待

    def _generate_ai_prompt(self, ai_id, current_task_for_ai, include_current_history=True):
        """
        特定のAI参加者に対するLLMへのプロンプト（指示文）を構築する。
        GDマネージャーがAIを「指揮」する中核。
        """
        system_instruction = (
            f"あなたはGDの参加者である{ai_id}です。あなたのペルソナは以下の通りです:\n"
            f"{self.participants[ai_id]['persona']}\n"
            "これまでのGD履歴と、あなたへの指示を踏まえ、適切にGDに参加する発言を生成してください。\n"
            "あなたの発言は、ペルソナに忠実であり、議論の目的達成に貢献するものでなければなりません。\n"
            "簡潔かつ自然な日本語で発言し、不要な説明やAIとしての言及は避けてください。"
        )

        messages = [{"role": "system", "content": system_instruction}]
        
        # 過去の会話履歴をプロンプトに含める（LLMにGDの文脈を伝えるため）
        if include_current_history:
            # 直近の履歴のみ含めるなど、トークン数と関連性を考慮して調整
            # ここではGDマネージャー自身のコメントは省く（システムとしての発言は含める）
            context_messages = [
                {"role": "user" if msg['speaker'] == "ユーザー" else "assistant", # LLMのロールに合わせて調整
                 "content": msg['content']}
                for msg in self.conversation_history if msg['speaker'] != "システム"
            ]
            messages.extend(context_messages)

        # 現在のGDフェーズや残り時間、AIへの具体的なタスクをLLMに伝える
        messages.append({"role": "user", "content": 
            f"現在のGDフェーズ: {self.current_phase}\n"
            f"残り時間: {int((self.time_limit_minutes * 60 - (time.time() - self.start_time)) / 60)}分\n"
            f"あなたのタスク: {current_task_for_ai}"
        })
        
        return messages

    def _get_ai_response(self, ai_id, task_for_ai, include_current_history=True):
        """
        特定のAI参加者（ai_id）の発言をLLMに生成させる。
        """
        messages = self._generate_ai_prompt(ai_id, task_for_ai, include_current_history)
        
        print(f" (GDマネージャー -> LLMへの指示 for {ai_id}): {task_for_ai[:50]}...") # デバッグ表示
        
        try:
            response = self.ai_client.chat_completions.create(
                model="gpt-4o", # 使用するLLMモデル。必要に応じてgpt-3.5-turboなどに変更。
                messages=messages,
                temperature=0.7, # 応答のランダム性
                max_tokens=150,  # 生成する最大トークン数
            )
            return response.choices[0].message.content.strip()
        except openai.APIConnectionError as e:
            print(f"エラー: OpenAI APIへの接続に失敗しました: {e}")
            return f"（{ai_id}）すみません、接続エラーのため発言できません。" 
        except openai.RateLimitError as e:
            print(f"エラー: OpenAI APIレート制限に達しました: {e}")
            return f"（{ai_id}）すみません、レート制限のため発言できません。"
        except openai.AuthenticationError as e:
            print(f"エラー: OpenAI API認証に失敗しました: {e}")
            return f"（{ai_id}）すみません、認証エラーのため発言できません。APIキーを確認してください。"
        except Exception as e:
            print(f"エラー: LLMの呼び出し中に予期せぬエラーが発生しました: {e}")
            return f"（{ai_id}）予期せぬエラーのため発言できません。"

    def process_user_input(self, user_text):
        """
        ユーザーからの入力を処理し、GDの進行を管理する。
        AIからの応答をトリガーする。
        """
        self.add_to_history("ユーザー", user_text) # ユーザーの発言を履歴に追加
        self.turn_count += 1
        self.current_speaker = "ユーザー" # 最新の発言者はユーザー

        print(f"\n[ユーザー]: {user_text}")

        # --- GDマネージャー（中央オーケストレーター）の主な判断ロジック ---
        # ここで、ユーザーのファシリテーション意図を分析し、GDフェーズを更新し、
        # 次にどのAIに参加させるかを決定します。
        
        # 1. ユーザーのファシリテーション意図の判断 (NLU) - 詳細な実装は後で
        # user_intent = self._analyze_user_intent(user_text) 
        # print(f"(デバッグ用) ユーザーの意図: {user_intent}")

        # 2. GDフェーズの進行判断 - 詳細な実装は後で
        # self._update_gd_phase(user_intent)

        # 3. 次に発言するAI参加者の決定とタスク付与
        # この部分は、実際のGDマネージャーの「知性」が現れる場所です。
        # 今回は簡易的に、ユーザー発言の後に全AIが順番に発言するシミュレーション
        ai_participants_to_respond = []
        for ai_name in ["AI_A", "AI_B", "AI_C"]: # 全てのAIに発言させる簡易的なルール
            ai_participants_to_respond.append(ai_name)
        
        # ユーザーが特定の質問をした場合に、そのAIが直接応答するロジックなどもここに追加
        # 例: if "AI_A" in user_text and "質問" in user_text: next_ai_id = "AI_A"; task = "質問に答える"

        # AIからの応答を順次生成・処理
        all_ai_responses_for_this_turn = []
        for ai_name in ai_participants_to_respond:
            task_for_ai = "直前の議論（ユーザーの発言も含む）を踏まえ、自身の役割とペルソナに沿って発言してください。"
            
            # --- ここでLLMを呼び出してAIの発言を生成 ---
            llm_response_text = self._get_ai_response(ai_name, task_for_ai)
            
            self.add_to_history(ai_name, llm_response_text) # 履歴に追加
            all_ai_responses_for_this_turn.append({"speaker": ai_name, "text": llm_response_text})
            
            # 実際はここでTTSを通じて音声を再生します
            print(f"[{ai_name}]: {llm_response_text}")
            time.sleep(1) # AIの発言間隔をシミュレート
            self.current_speaker = ai_name # 発言者を更新

        # GDの終了条件判定など
        if time.time() - self.start_time > self.time_limit_minutes * 60:
            print("\n--- GD終了: 制限時間になりました ---")
            return False # GDを終了する
        return True # GDを続行する

    # --- 簡易フィードバックの生成 (GD終了後に行う) ---
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

        # 3. GDフェーズの最終到達点
        report["最終到達フェーズ"] = self.current_phase

        # 4. LLMによる簡単な振り返り（より高度な分析は後で）
        # 全会話履歴をLLMに送り、簡単な要約や示唆を求める
        feedback_prompt = (
            "以下のGDログについて、ファシリテーター（ユーザー）の役割を中心に、議論の進行状況や雰囲気に関する簡単な振り返り（50文字程度）と、改善点に関する短い示唆（50文字程度）を生成してください。\n\n"
            + "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in self.conversation_history])
        )
        try:
            llm_feedback = self.ai_client.chat_completions.create(
                model="gpt-3.5-turbo", # フィードバック生成にはgpt-3.5-turboで十分な場合が多い
                messages=[{"role": "system", "content": "あなたはGDのパフォーマンスを評価し、簡潔な振り返りと示唆を提供するアシスタントです。"},
                          {"role": "user", "content": feedback_prompt}],
                temperature=0.5,
                max_tokens=150
            ).choices[0].message.content.strip()
            report["LLMからの総括と示唆"] = llm_feedback
        except Exception as e:
            report["LLMからの総括と示唆"] = f"フィードバック生成中にエラー: {e}"

        print("\n--- GD簡易フィードバックレポート ---")
        for key, value in report.items():
            print(f"{key}: {value}")
        print("\nレポート生成が完了しました。")
        return report

# --- スクリプトの実行（GDManagerの動作確認用） ---
if __name__ == "__main__":
    print("--- GDManagerの統合テスト開始 ---")
    manager = GDManager() # GDManagerを初期化

    # GDのメインループをシミュレート
    running_gd = True
    while running_gd:
        try:
            user_input = input("\nあなた（ファシリテーター）の発言を入力してください（終了するには '終了' と入力）: ")
            if user_input.lower() == '終了':
                running_gd = False
                break
            
            running_gd = manager.process_user_input(user_input)

        except KeyboardInterrupt: # Ctrl+Cで中断された場合
            print("\nGDが中断されました。")
            running_gd = False
            
        if not running_gd:
            break

    # GD終了後にフィードバックレポートを生成
    final_report = manager.generate_simple_feedback_report()
    print("\nGDシミュレーションが完了しました。")