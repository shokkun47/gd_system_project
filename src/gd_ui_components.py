"""
Zoom風GD UIコンポーネント
4画面構成: ユーザー名入力 → テーマ思考 → GD進行 → フィードバック
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QTextEdit, QFrame, QStackedWidget, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, Signal as pyqtSignal, QTimer
from PySide6.QtGui import QPixmap, QFont
import os

# アバターヘルパーをインポート
try:
    from avatar_helper import get_avatar_path, get_participant_color, PARTICIPANT_COLORS
except ImportError:
    # フォールバック: 色定義のみ
    PARTICIPANT_COLORS = {
        "user": "#3498db",
        "ai_1": "#e74c3c",
        "ai_2": "#2ecc71",
        "ai_3": "#9b59b6",
        "ai_4": "#e67e22",
    }
    def get_participant_color(p_type):
        return PARTICIPANT_COLORS.get(p_type, "#95a5a6")
    def get_avatar_path(p_type):
        return ""

class ParticipantAvatar(QWidget):
    """参加者アバター表示ウィジェット"""
    
    def __init__(self, name, participant_type, parent=None):
        super().__init__(parent)
        self.name = name
        self.participant_type = participant_type
        self.color = get_participant_color(participant_type)
        self.is_speaking = False
        
        # レイアウト
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(1)  # スペースを削減（2→1）
        layout.setContentsMargins(0, 0, 0, 0)
        
        # ウィジェット全体のスタイル（下線を確実に削除）
        self.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        
        # アバター画像（真四角）
        from PySide6.QtGui import QPainter, QBrush, QColor, QFont, QPen
        
        # 枠線を含めた四角形画像を生成
        avatar_size = 64
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(avatar_size, avatar_size)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        
        # 四角形アバター画像を生成（枠線込み）
        pixmap = QPixmap(avatar_size, avatar_size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 外側の枠線（四角）
        painter.setPen(QPen(QColor(self.color), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(1, 1, avatar_size-3, avatar_size-3)
        
        # 内側の塗りつぶし四角
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(self.color)))
        painter.drawRect(3, 3, avatar_size-6, avatar_size-6)
        
        # 頭文字を中央に描画
        painter.setPen(QColor("#ffffff"))
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        initial = name[0].upper() if name else "?"
        # Qt.AlignCenter | Qt.AlignVCenter で水平・垂直両方の中央揃え
        painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.AlignVCenter, initial)
        
        painter.end()
        self.avatar_label.setPixmap(pixmap)
        
        # ラベル自体には枠線を設定しない（画像に描画済み）
        self.avatar_label.setStyleSheet("QLabel { border: none; background-color: transparent; }")
        
        # コンテナ（枠線なし、サイズをアバターと同じに）
        self.frame = QWidget()
        self.frame.setFixedSize(avatar_size, avatar_size)  # アバターと同じサイズに
        self.frame.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        
        frame_layout = QVBoxLayout()
        frame_layout.setAlignment(Qt.AlignCenter)
        frame_layout.setContentsMargins(0, 0, 0, 0)  # マージンをなくす
        frame_layout.setSpacing(0)
        frame_layout.addWidget(self.avatar_label)
        self.frame.setLayout(frame_layout)
        
        # 名前ラベル（小さく、下線なし）
        self.name_label = QLabel(name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("""
            font-size: 10px; 
            font-weight: bold;
            text-decoration: none;
            border: none;
            background-color: transparent;
            margin: 0px;
            padding: 0px;
        """)
        
        # 役職ラベル（新規追加）
        self.role = None
        self.role_label = QLabel("")
        self.role_label.setAlignment(Qt.AlignCenter)
        self.role_label.setFixedHeight(12)  # 固定高さを設定（役職の有無に関わらず）
        self.role_label.setStyleSheet("""
            font-size: 9px;
            color: #000000;
            font-style: italic;
            border: none;
            background-color: transparent;
            margin: 0px;
            padding: 0px;
        """)
        
        # 発言中ラベル（小さく、常に表示）
        self.speaking_label = QLabel("🔊")
        self.speaking_label.setAlignment(Qt.AlignCenter)
        self.speaking_label.setStyleSheet("font-size: 14px; margin: 0px; padding: 0px;")
        # 透明度エフェクトを設定（初期状態は透明）
        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(0.0)
        self.speaking_label.setGraphicsEffect(self.opacity_effect)
        
        layout.addWidget(self.frame)
        layout.addWidget(self.name_label)
        layout.addWidget(self.role_label)  # 役職ラベルを追加
        layout.addWidget(self.speaking_label)
        
        self.setLayout(layout)
    
    def set_speaking(self, speaking):
        """発言中状態を設定（画像を再描画）"""
        from PySide6.QtGui import QPainter, QBrush, QColor, QFont, QPen
        
        self.is_speaking = speaking
        avatar_size = 64
        border_width = 4 if speaking else 2
        
        # アバター画像を再描画（四角形）
        pixmap = QPixmap(avatar_size, avatar_size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 外側の枠線（四角）
        painter.setPen(QPen(QColor(self.color), border_width))
        painter.setBrush(Qt.NoBrush)
        half_border = border_width // 2
        painter.drawRect(half_border, half_border, 
                        avatar_size - border_width, avatar_size - border_width)
        
        # 内側の塗りつぶし四角
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(self.color)))
        margin = border_width + 1
        painter.drawRect(margin, margin, avatar_size - margin*2, avatar_size - margin*2)
        
        # 頭文字を中央に描画
        painter.setPen(QColor("#ffffff"))
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        initial = self.name[0].upper() if self.name else "?"
        # pixmap.rect()で完全な中央揃え
        painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.AlignVCenter, initial)
        
        painter.end()
        self.avatar_label.setPixmap(pixmap)
        
        # 発言中ラベルの透明度を切り替え
        if speaking:
            self.opacity_effect.setOpacity(1.0)  # 不透明
        else:
            self.opacity_effect.setOpacity(0.0)  # 透明
    
    def update_role(self, role):
        """役職を更新（固定高さを維持）"""
        self.role = role
        if role:
            self.role_label.setText(f"{role}")
        else:
            self.role_label.setText("")  # 空文字でも固定高さを維持


class UserInputScreen(QWidget):
    """画面1: ユーザー名入力（名字と名前を別々に入力）"""
    system_start_requested = pyqtSignal(str, str)  # 名字と名前を送信
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # タイトル
        title = QLabel("グループディスカッションシミュレーター")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 30px;")
        title.setAlignment(Qt.AlignCenter)
        
        # 説明
        description = QLabel("名字と名前を入力してください")
        description.setStyleSheet("font-size: 16px; margin-bottom: 20px;")
        description.setAlignment(Qt.AlignCenter)
        
        # 名字入力フィールド
        lastname_label = QLabel("<span style='font-weight: bold;'>名字</span>")
        lastname_label.setStyleSheet("font-size: 14px; margin-bottom: 5px;")
        lastname_label.setAlignment(Qt.AlignCenter)
        lastname_label.setTextFormat(Qt.RichText)
        
        self.lastname_input = QLineEdit()
        self.lastname_input.setPlaceholderText("例: 山田")
        self.lastname_input.setStyleSheet("""
            QLineEdit {
                font-size: 18px;
                padding: 10px;
                border: 2px solid #3498db;
                border-radius: 5px;
                max-width: 400px;
            }
        """)
        self.lastname_input.setMaximumWidth(400)
        self.lastname_input.returnPressed.connect(lambda: self.firstname_input.setFocus())
        
        # 名前入力フィールド
        firstname_label = QLabel("<span style='font-weight: bold;'>名前</span>")
        firstname_label.setStyleSheet("font-size: 14px; margin-bottom: 5px; margin-top: 15px;")
        firstname_label.setAlignment(Qt.AlignCenter)
        firstname_label.setTextFormat(Qt.RichText)
        
        self.firstname_input = QLineEdit()
        self.firstname_input.setPlaceholderText("例: 太郎")
        self.firstname_input.setStyleSheet("""
            QLineEdit {
                font-size: 18px;
                padding: 10px;
                border: 2px solid #3498db;
                border-radius: 5px;
                max-width: 400px;
            }
        """)
        self.firstname_input.setMaximumWidth(400)
        self.firstname_input.returnPressed.connect(self._on_start_clicked)
        
        # ボタン
        self.start_button = QPushButton("システム開始")
        self.start_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 30px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                max-width: 400px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.start_button.setMaximumWidth(400)
        self.start_button.clicked.connect(self._on_start_clicked)
        
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(lastname_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.lastname_input, alignment=Qt.AlignCenter)
        layout.addWidget(firstname_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.firstname_input, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _on_start_clicked(self):
        lastname = self.lastname_input.text().strip()
        firstname = self.firstname_input.text().strip()
        if lastname and firstname:
            self.system_start_requested.emit(lastname, firstname)
        elif not lastname:
            self.lastname_input.setPlaceholderText("⚠ 名字を入力してください")
        elif not firstname:
            self.firstname_input.setPlaceholderText("⚠ 名前を入力してください")


class ThinkingScreen(QWidget):
    """画面2: テーマ表示 + 思考時間"""
    gd_start_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # テーマラベル
        theme_title = QLabel("【GDテーマ】")
        theme_title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 15px;")
        theme_title.setAlignment(Qt.AlignCenter)
        
        self.theme_text = QLabel()
        self.theme_text.setWordWrap(True)
        self.theme_text.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                border: 2px solid #3498db;
            }
        """)
        self.theme_text.setAlignment(Qt.AlignCenter)
        self.theme_text.setMaximumWidth(800)
        
        # カウントダウンラベル
        self.countdown_label = QLabel()
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #e74c3c;
                margin-top: 30px;
            }
        """)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        
        # メッセージラベル
        self.message_label = QLabel("1分間、テーマについて考える時間です")
        self.message_label.setStyleSheet("font-size: 16px; margin-top: 20px;")
        self.message_label.setAlignment(Qt.AlignCenter)
        
        layout.addStretch()
        layout.addWidget(theme_title)
        layout.addWidget(self.theme_text, alignment=Qt.AlignCenter)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.message_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_countdown)
        self.remaining_seconds = 0
    
    def start_thinking(self, theme, seconds=60):
        """思考時間を開始"""
        self.theme_text.setText(theme)
        self.remaining_seconds = seconds
        self._update_countdown()
        self.timer.start(1000)
    
    def _update_countdown(self):
        self.countdown_label.setText(f"{self.remaining_seconds}秒")
        if self.remaining_seconds <= 0:
            self.timer.stop()
            self.message_label.setText("まもなくGDが開始されます...")
            # 2秒後にGD開始
            QTimer.singleShot(2000, self.gd_start_requested.emit)
        else:
            self.remaining_seconds -= 1


class GDScreen(QWidget):
    """画面3: GD進行中（Zoom風レイアウト）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 余白をなくす
        
        # 背景色を白に設定
        self.setStyleSheet("background-color: white;")
        
        # システム発言中バナー（初期状態は非表示）
        self.system_banner = QLabel("🔊 システム発言中...")
        self.system_banner.setAlignment(Qt.AlignCenter)
        self.system_banner.setStyleSheet("""
            QLabel {
                background-color: #d1ecf1;
                color: #0c5460;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.system_banner.hide()  # 初期状態は非表示
        
        # AI思考中/発言中バナー（初期状態は非表示）
        self.ai_status_banner = QLabel("")
        self.ai_status_banner.setAlignment(Qt.AlignCenter)
        self.ai_status_banner.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                color: #856404;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.ai_status_banner.hide()  # 初期状態は非表示
        
        # 上部: アバターと残り時間を横並びに配置
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左側: アバターエリア
        avatar_container = QFrame()
        avatar_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 3px;
            }
        """)
        avatar_container.setMaximumHeight(120)  # 高さを制限（150→120）
        avatar_layout = QHBoxLayout()
        avatar_layout.setSpacing(8)  # スペースを削減（15→8）
        avatar_layout.setAlignment(Qt.AlignCenter)
        avatar_layout.setContentsMargins(5, 3, 5, 3)  # マージンを削減
        
        # アバターを作成（5人分）- 初期状態
        self.avatars = {}
        self.avatar_layout = avatar_layout
        self.avatar_container = avatar_container
        
        # 初期プレースホルダー
        participant_info = [
            ("ユーザー", "user"),
            ("AI参加者1", "ai_1"),
            ("AI参加者2", "ai_2"),
            ("AI参加者3", "ai_3"),
            ("AI参加者4", "ai_4"),
        ]
        
        for name, p_type in participant_info:
            avatar = ParticipantAvatar(name, p_type)
            self.avatars[name] = avatar
            avatar_layout.addWidget(avatar)
        
        avatar_container.setLayout(avatar_layout)
        
        # 右側: 残り時間表示エリア
        timer_container = QFrame()
        timer_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        timer_container.setFixedSize(150, 120)  # 固定サイズに変更
        timer_layout = QVBoxLayout()
        timer_layout.setAlignment(Qt.AlignCenter)
        timer_layout.setContentsMargins(5, 5, 5, 5)
        timer_layout.setSpacing(0)  # 余白を0に変更
        
        timer_title = QLabel("残り時間")
        timer_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                border: none;
                background-color: transparent;
            }
        """)
        timer_title.setAlignment(Qt.AlignCenter)
        
        self.timer_label = QLabel("00:00")  # 初期値は0:00（GDManagerから設定される）
        self.timer_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #27ae60;
                border: none;
                background-color: transparent;
            }
        """)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setMinimumHeight(50)  # 最小高さを設定
        
        timer_layout.addStretch()
        timer_layout.addWidget(timer_title)
        timer_layout.addWidget(self.timer_label)
        timer_layout.addStretch()
        timer_container.setLayout(timer_layout)
        
        # アバターと残り時間を横並びに配置
        top_layout.addWidget(avatar_container, stretch=1)
        top_layout.addWidget(timer_container)
        
        # 下部: 議事録エリア
        self.minutes_label = QLabel("📋 テーマ")
        self.minutes_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                margin: 5px 10px;
            }
        """)
        
        self.minutes_text = QTextEdit()
        self.minutes_text.setReadOnly(True)
        self.minutes_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        
        # ローディングオーバーレイ（初期状態は非表示）
        self.loading_overlay = QFrame()
        self.loading_overlay.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.9);
                border: none;
            }
        """)
        loading_layout = QVBoxLayout()
        loading_layout.setAlignment(Qt.AlignCenter)
        
        loading_spinner = QLabel("⏳")
        loading_spinner.setStyleSheet("""
            QLabel {
                font-size: 48px;
                background-color: transparent;
            }
        """)
        loading_spinner.setAlignment(Qt.AlignCenter)
        
        loading_text = QLabel("GDを開始しています...")
        loading_text.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                background-color: transparent;
                margin-top: 10px;
            }
        """)
        loading_text.setAlignment(Qt.AlignCenter)
        
        loading_layout.addWidget(loading_spinner)
        loading_layout.addWidget(loading_text)
        self.loading_overlay.setLayout(loading_layout)
        self.loading_overlay.hide()  # 初期状態は非表示
        
        main_layout.addWidget(self.system_banner)
        main_layout.addWidget(self.ai_status_banner)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.minutes_label)
        main_layout.addWidget(self.minutes_text, stretch=1)
        
        self.setLayout(main_layout)
        
        # ローディングオーバーレイを最前面に配置（レイアウトの外に配置）
        self.loading_overlay.setParent(self)
        self.loading_overlay.setGeometry(0, 0, self.width(), self.height())
        self.loading_overlay.raise_()  # 最前面に表示
    
    def set_participants(self, participant_names):
        """
        参加者名を設定してアバターを更新
        
        Args:
            participant_names: {表示名: タイプ} の辞書
                例: {"山田": "user", "田中": "ai_1", "佐藤": "ai_2", ...}
        """
        # 既存のアバターをクリア
        for avatar in self.avatars.values():
            avatar.setParent(None)
            avatar.deleteLater()
        self.avatars.clear()
        
        # 新しいアバターを作成
        for name, p_type in participant_names.items():
            avatar = ParticipantAvatar(name, p_type)
            self.avatars[name] = avatar
            self.avatar_layout.addWidget(avatar)
    
    def update_speaker(self, speaker_name):
        """発言者を更新（空文字列の場合は音声マークを消す）"""
        if not speaker_name or speaker_name == "":
            # 誰もしゃべっていない時は音声マークを消す
            for name, avatar in self.avatars.items():
                avatar.set_speaking(False)
        else:
            for name, avatar in self.avatars.items():
                avatar.set_speaking(name == speaker_name)
    
    def update_participant_role(self, participant_name, role):
        """参加者の役職を更新"""
        if participant_name in self.avatars:
            self.avatars[participant_name].update_role(role)
    
    def update_timer(self, remaining_minutes, remaining_seconds):
        """残り時間を更新"""
        self.timer_label.setText(f"{remaining_minutes:02d}:{remaining_seconds:02d}")
        
        # 残り時間が5分以下になったら色を変更
        if remaining_minutes < 5:
            self.timer_label.setStyleSheet("""
                QLabel {
                    font-size: 36px;
                    font-weight: bold;
                    color: #c0392b;
                    border: none;
                    background-color: transparent;
                }
            """)
        elif remaining_minutes < 10:
            self.timer_label.setStyleSheet("""
                QLabel {
                    font-size: 36px;
                    font-weight: bold;
                    color: #e67e22;
                    border: none;
                    background-color: transparent;
                }
            """)
        else:
            self.timer_label.setStyleSheet("""
                QLabel {
                    font-size: 36px;
                    font-weight: bold;
                    color: #27ae60;
                    border: none;
                    background-color: transparent;
                }
            """)
    
    def update_minutes(self, minutes_text):
        """議事録を更新"""
        self.minutes_text.setPlainText(minutes_text)
        # 自動スクロール
        from PySide6.QtGui import QTextCursor
        cursor = self.minutes_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.minutes_text.setTextCursor(cursor)
    
    def show_system_speaking(self):
        """システム発言中バナーを表示"""
        self.system_banner.show()
    
    def hide_system_speaking(self):
        """システム発言中バナーを非表示"""
        self.system_banner.hide()
    
    def show_ai_thinking(self, ai_name):
        """AI思考中バナーを表示"""
        self.ai_status_banner.setText(f"💭 {ai_name}さんが考えています...")
        self.ai_status_banner.show()
    
    def show_ai_speaking(self, ai_name):
        """AI発言中バナーを表示"""
        self.ai_status_banner.setText(f"🔊 {ai_name}さんが話しています...")
        self.ai_status_banner.show()
    
    def hide_ai_status(self):
        """AI状態バナーを非表示"""
        self.ai_status_banner.hide()
    
    def set_theme(self, theme):
        """テーマラベルにテーマタイトルのみを表示"""
        theme_title = theme.splitlines()[0] if theme else ""
        self.minutes_label.setText(f"📋 テーマ - {theme_title}")
    
    def show_loading(self):
        """ローディングオーバーレイを表示"""
        self.loading_overlay.setGeometry(0, 0, self.width(), self.height())
        self.loading_overlay.show()
        self.loading_overlay.raise_()  # 最前面に表示
    
    def hide_loading(self):
        """ローディングオーバーレイを非表示"""
        self.loading_overlay.hide()
    
    def resizeEvent(self, event):
        """ウィンドウリサイズ時にローディングオーバーレイのサイズも更新"""
        super().resizeEvent(event)
        if self.loading_overlay.isVisible():
            self.loading_overlay.setGeometry(0, 0, self.width(), self.height())


class GroupSelectionScreen(QWidget):
    """実験群/統制群選択画面"""
    group_selected = pyqtSignal(str)  # "experimental" または "control" を送信
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # タイトル
        title = QLabel("実験群/統制群の選択")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 30px;")
        title.setAlignment(Qt.AlignCenter)
        
        # 説明
        description = QLabel("どちらのグループに参加しますか？")
        description.setStyleSheet("font-size: 16px; margin-bottom: 30px;")
        description.setAlignment(Qt.AlignCenter)
        
        # ボタンレイアウト
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # 実験群ボタン
        self.experimental_button = QPushButton("実験群")
        self.experimental_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                padding: 20px 40px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.experimental_button.clicked.connect(lambda: self.group_selected.emit("experimental"))
        
        # 統制群ボタン
        self.control_button = QPushButton("統制群")
        self.control_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                padding: 20px 40px;
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 10px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.control_button.clicked.connect(lambda: self.group_selected.emit("control"))
        
        button_layout.addWidget(self.experimental_button)
        button_layout.addWidget(self.control_button)
        
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)


class GDStartConfirmScreen(QWidget):
    """GD開始確認画面（警告表示）"""
    confirmed = pyqtSignal()  # 確認ボタンが押されたときに発火
    cancelled = pyqtSignal()  # キャンセルボタンが押されたときに発火
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # 警告アイコンとメッセージ
        warning_label = QLabel("⚠️")
        warning_label.setStyleSheet("font-size: 64px; margin-bottom: 20px;")
        warning_label.setAlignment(Qt.AlignCenter)
        
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 20px;
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 10px;
                max-width: 600px;
            }
        """)
        self.message_label.setAlignment(Qt.AlignCenter)
        
        # ボタンレイアウト
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # 開始ボタン
        self.confirm_button = QPushButton("開始する")
        self.confirm_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 30px;
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.confirm_button.clicked.connect(self.confirmed.emit)
        
        # キャンセルボタン
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 30px;
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.cancel_button.clicked.connect(self.cancelled.emit)
        
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addStretch()
        layout.addWidget(warning_label)
        layout.addWidget(self.message_label, alignment=Qt.AlignCenter)
        layout.addSpacing(30)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def set_message(self, message):
        """警告メッセージを設定"""
        self.message_label.setText(message)


class FeedbackScreen(QWidget):
    """画面4: フィードバック表示"""
    next_gd_requested = pyqtSignal()  # 2回目GD開始用のシグナル（実験群のみ）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # タイトル
        title = QLabel("📊 フィードバックレポート")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        
        # 進捗表示ラベル（初期状態は非表示）
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #3498db;
                font-weight: bold;
                padding: 10px;
                background-color: #e8f4f8;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()  # 初期状態は非表示
        
        # フィードバック表示エリア
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                font-size: 14px;
            }
        """)
        
        # ボタンエリア
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # 2回目GD開始ボタン（実験群のみ表示）
        self.next_gd_button = QPushButton("2回目のグループディスカッションを開始する")
        self.next_gd_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 30px;
                background-color: #e67e22;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.next_gd_button.clicked.connect(self.next_gd_requested.emit)
        self.next_gd_button.hide()  # 初期状態は非表示
        
        button_layout.addStretch()
        button_layout.addWidget(self.next_gd_button)
        button_layout.addStretch()
        
        layout.addWidget(title)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.feedback_text)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def set_feedback(self, feedback_dict, show_next_button=False):
        """フィードバックを設定"""
        # 進捗表示を非表示
        self.progress_label.hide()
        
        # 実験群用フィードバックのみを表示（会話ログなどは表示しない）
        feedback_md = "# GDフィードバックレポート（実験群用）\n\n"
        
        # 1. 採点サマリー（5項目）
        scores = feedback_dict.get("ファシリテーション手法スコア", {})
        total = feedback_dict.get("合計スコア", "")
        if scores:
            feedback_md += "## ファシリテーション手法スコア\n\n"
            for item_key in ["目的確認", "役割分担", "意見引き出し", "議論整理", "時間管理"]:
                if item_key in scores:
                    feedback_md += f"- **{item_key}**: {scores[item_key]}点\n"
            if total:
                feedback_md += f"\n**合計スコア**: {total}\n\n"
            feedback_md += "---\n\n"
        
        # 2. 実験群用フィードバック本文
        exp_feedback = feedback_dict.get("実験群用フィードバック", "")
        if exp_feedback:
            feedback_md += "## フィードバック（Good / More / Action）\n\n"
            feedback_md += exp_feedback.strip() + "\n"
        
        # Markdownとしてレンダリング
        self.feedback_text.setMarkdown(feedback_md)
        
        # 2回目GD開始ボタンの表示/非表示を制御
        if show_next_button:
            self.next_gd_button.show()
        else:
            self.next_gd_button.hide()
    
    def show_progress(self, message):
        """進捗メッセージを表示"""
        self.progress_label.setText(message)
        self.progress_label.show()
        # フィードバックテキストをクリア
        self.feedback_text.setMarkdown("")


class ControlGroupAfterFirstScreen(QWidget):
    """統制群用: 1回目終了後の画面（学習用ドキュメント表示 + 2回目GD開始ボタン）"""
    next_gd_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # メッセージ
        message = QLabel("1回目のグループディスカッションが終了しました。\n\n"
                         "統制群の方は、以下の「ファシリテーション・ハンドブック」を読んでください。\n"
                         "内容を確認したら、2回目のグループディスカッションを開始してください。")
        message.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        message.setAlignment(Qt.AlignLeft)
        
        # 学習用ドキュメント表示エリア（統制群用資料）
        self.doc_text = QTextEdit()
        self.doc_text.setReadOnly(True)
        self.doc_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        # Markdownで強調表現を付けた統制群用ドキュメント
        self.doc_text.setMarkdown(
            "# グループディスカッションを成功に導く**5つのファシリテーション手法**\n\n"
            "以下の5つの基本動作を意識し、**「声に出して」実践**することで、議論の質は大きく高まります。\n\n"
            "---\n\n"
            "## 1. 目的の確認\n\n"
            "議論が迷走しないよう、開始直後に**「何を決める場なのか」**を全員で共有しましょう。\n\n"
            "- **ポイント**: 議論の冒頭で、ゴールや議題を明確に宣言する。\n\n"
            "- **使えるフレーズ**:\n\n"
            "  - 「**今日の議論のゴール**は、○○を決定することですね」\n\n"
            "  - 「まずは、**○○について話し合って**いきましょう」\n\n"
            "---\n\n"
            "## 2. 役割分担\n\n"
            "円滑な進行と記録のために、メンバーに**役割**を割り振りましょう。\n\n"
            "- **ポイント**: 「書記」や「タイムキーパー」などの役割を、**具体的に指名して依頼**する。\n\n"
            "- **使えるフレーズ**:\n\n"
            "  - 「**役割分担を決めましょう**」\n\n"
            "  - 「Aさん、**書記をお願い**できますか？」\n\n"
            "  - 「Bさん、**タイムキーパーをお願い**してもいいですか？」\n\n"
            "---\n\n"
            "## 3. 意見の引き出し\n\n"
            "全員が発言しやすい環境を作りましょう。特に、**発言が少ない人への配慮**が重要です。\n\n"
            "- **ポイント**: 特定の人を指名したり、全体に問いかけたりして、発言を促す。\n\n"
            "- **使えるフレーズ**:\n\n"
            "  - 「**Cさんは、この点についてどう思いますか？**」（指名）\n\n"
            "  - 「**他の方で、違う意見を持っている人はいますか？**」（全体）\n\n"
            "---\n\n"
            "## 4. 議論の整理\n\n"
            "意見が出っぱなしにならないよう、適度なタイミングで**要約・整理**しましょう。\n\n"
            "- **ポイント**: 出てきた意見を要約したり、共通点・対立点を整理して伝える。\n\n"
            "- **使えるフレーズ**:\n\n"
            "  - 「**ここまでの意見をまとめると**、○○案と××案が出ていますね」\n\n"
            "  - 「**つまり**、Aさんの意見は〜〜ということですね」\n\n"
            "---\n\n"
            "## 5. 時間管理\n\n"
            "限られた時間内で結論を出すために、常に**残り時間を意識して共有**しましょう。\n\n"
            "- **ポイント**: 残り時間をアナウンスし、次のステップ（まとめ等）への移行を促す。\n\n"
            "- **使えるフレーズ**:\n\n"
            "  - 「**残り5分です。そろそろ意見をまとめていきましょう**」\n\n"
            "  - 「**時間が半分過ぎました。次の議題に移りましょうか**」\n\n"
            "---\n\n"
            "## ★ アドバイス\n\n"
            "ファシリテーションは「**慣れ**」も重要ですが、まずはこれらの基本動作を**恐れずに発言してみる**ことが上達への第一歩です。\n\n"
            "次回の議論では、この5つを**最低1回ずつは使ってみる**つもりで取り組んでみてください。\n\n"
        )
        
        # 2回目GD開始ボタン
        self.next_gd_button = QPushButton("2回目のグループディスカッションを開始する")
        self.next_gd_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 15px 40px;
                background-color: #e67e22;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.next_gd_button.clicked.connect(self.next_gd_requested.emit)
        
        layout.addWidget(message)
        layout.addWidget(self.doc_text)
        layout.addWidget(self.next_gd_button, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)

