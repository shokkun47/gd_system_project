"""
Zoom風GD UIコンポーネント
4画面構成: ユーザー名入力 → テーマ思考 → GD進行 → フィードバック
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QTextEdit, QFrame, QStackedWidget, QGraphicsOpacityEffect,
    QSlider, QProgressBar
)
from PySide6.QtCore import Qt, Signal as pyqtSignal, QTimer
from PySide6.QtGui import QPixmap, QFont
import os
import threading

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
        
        # AI思考中/発言中/待機中バナー（常に表示）
        self.ai_status_banner = QLabel("⏳ 待機中...")
        self.ai_status_banner.setAlignment(Qt.AlignCenter)
        self.ai_status_banner.setStyleSheet("""
            QLabel {
                background-color: #e8f4f8;
                color: #0c5460;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        # 初期状態から表示（常に表示）
        
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
        self.ai_status_banner.show()
    
    def show_ai_speaking(self, ai_name):
        """AI発言中バナーを表示"""
        self.ai_status_banner.setText(f"🔊 {ai_name}さんが話しています...")
        self.ai_status_banner.setStyleSheet("""
            QLabel {
                background-color: #d4edda;
                color: #155724;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.ai_status_banner.show()
    
    def show_user_speaking(self, user_name):
        """ユーザー発言中バナーを表示"""
        self.ai_status_banner.setText(f"🎤 {user_name}さんが話しています...")
        self.ai_status_banner.setStyleSheet("""
            QLabel {
                background-color: #cce5ff;
                color: #004085;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #99ccff;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.ai_status_banner.show()
    
    def show_waiting(self):
        """待機中バナーを表示"""
        self.ai_status_banner.setText("⏳ 待機中...")
        self.ai_status_banner.setStyleSheet("""
            QLabel {
                background-color: #e8f4f8;
                color: #0c5460;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.ai_status_banner.show()
    
    def hide_ai_status(self):
        """AI状態バナーを待機中に戻す（非表示ではなく待機中を表示）"""
        self.show_waiting()
    
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
    cancelled = pyqtSignal()  # キャンセルボタンが押されたときに発火（互換性のため残す）
    thinking_timeout = pyqtSignal()  # 思考時間終了時に発火
    
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
        
        # 思考時間カウントダウンラベル
        self.countdown_label = QLabel()
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #e74c3c;
                margin-top: 20px;
                margin-bottom: 10px;
            }
        """)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.hide()  # 初期状態は非表示
        
        # 思考時間メッセージラベル
        self.thinking_message_label = QLabel()
        self.thinking_message_label.setStyleSheet("font-size: 16px; margin-bottom: 20px;")
        self.thinking_message_label.setAlignment(Qt.AlignCenter)
        self.thinking_message_label.hide()  # 初期状態は非表示
        
        # アナウンスメッセージラベル
        self.announcement_label = QLabel()
        self.announcement_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #27ae60;
                font-weight: bold;
                padding: 10px;
                background-color: #d5f4e6;
                border: 1px solid #27ae60;
                border-radius: 5px;
                margin-bottom: 20px;
            }
        """)
        self.announcement_label.setAlignment(Qt.AlignCenter)
        self.announcement_label.setWordWrap(True)
        self.announcement_label.hide()  # 初期状態は非表示
        
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
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.confirm_button.clicked.connect(self.confirmed.emit)
        
        button_layout.addWidget(self.confirm_button)
        
        layout.addStretch()
        layout.addWidget(warning_label)
        layout.addWidget(self.message_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.thinking_message_label)
        layout.addWidget(self.announcement_label)
        layout.addSpacing(30)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # 思考時間タイマー
        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self._update_thinking_countdown)
        self.remaining_seconds = 0
        self.thinking_active = False
    
    def set_message(self, message):
        """警告メッセージを設定"""
        self.message_label.setText(message)
        # 画面表示時に「開始する」ボタンを即座に無効化
        self.confirm_button.setEnabled(False)
    
    def start_thinking_time(self, seconds=120):
        """思考時間を開始（2分間、開発モードの場合は10秒）"""
        self.remaining_seconds = seconds
        self.thinking_active = True
        self.countdown_label.show()
        self.thinking_message_label.hide()  # メッセージラベルを非表示（文字を消す）
        self.confirm_button.setEnabled(False)  # 思考時間中は開始ボタンを無効化
        self.announcement_label.hide()
        self._update_thinking_countdown()
        self.thinking_timer.start(1000)  # 1秒ごとに更新
    
    def _update_thinking_countdown(self):
        """思考時間カウントダウンを更新"""
        if self.remaining_seconds > 0:
            minutes = self.remaining_seconds // 60
            seconds = self.remaining_seconds % 60
            self.countdown_label.setText(f"{minutes:02d}:{seconds:02d}")
            self.remaining_seconds -= 1
        else:
            # 思考時間終了
            self.thinking_timer.stop()
            self.thinking_active = False
            self.countdown_label.hide()
            self.thinking_message_label.hide()
            self.announcement_label.setText("思考時間が終了しました。「開始する」ボタンを押して、グループディスカッションを開始してください。")
            self.announcement_label.show()
            # ボタンはまだ有効化しない（アナウンス終了後に有効化）
            self.confirm_button.setEnabled(False)
            # アナウンスシグナルを発火
            self.thinking_timeout.emit()
    
    def enable_confirm_button_after_announcement(self):
        """アナウンス再生後に開始ボタンを有効化"""
        self.confirm_button.setEnabled(True)
    
    def stop_thinking_time(self):
        """思考時間を停止"""
        if self.thinking_timer.isActive():
            self.thinking_timer.stop()
        self.thinking_active = False
        self.countdown_label.hide()
        self.thinking_message_label.hide()
        self.announcement_label.hide()
        self.confirm_button.setEnabled(True)


class FeedbackScreen(QWidget):
    """画面4: フィードバック表示"""
    next_gd_requested = pyqtSignal()  # 2回目GD開始用のシグナル（実験群のみ）
    reading_timeout = pyqtSignal()  # 読書時間終了時に発火
    
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
        
        # 読書時間カウントダウンラベル
        self.reading_countdown_label = QLabel()
        self.reading_countdown_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #e74c3c;
                margin-bottom: 10px;
            }
        """)
        self.reading_countdown_label.setAlignment(Qt.AlignCenter)
        self.reading_countdown_label.hide()  # 初期状態は非表示
        
        # 読書時間メッセージラベル
        self.reading_message_label = QLabel()
        self.reading_message_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        self.reading_message_label.setAlignment(Qt.AlignCenter)
        self.reading_message_label.hide()  # 初期状態は非表示
        
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
        
        layout.addWidget(title)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.reading_countdown_label)
        layout.addWidget(self.reading_message_label)
        layout.addWidget(self.feedback_text)
        
        self.setLayout(layout)
        
        # 読書時間タイマー
        self.reading_timer = QTimer()
        self.reading_timer.timeout.connect(self._update_reading_countdown)
        self.reading_remaining_seconds = 0
        self.reading_active = False
    
    def start_reading_time(self, seconds=300):
        """読書時間を開始（5分間、開発モードの場合は10秒）"""
        self.reading_remaining_seconds = seconds
        self.reading_active = True
        self.reading_countdown_label.show()
        self.reading_message_label.show()
        # 時間に応じてメッセージを変更（アナウンス文言に合わせる）
        if seconds <= 30:
            self.reading_message_label.setText("AIからのフィードバックレポートを10秒間読み、2回目のグループディスカッションに備えてください。")
        else:
            self.reading_message_label.setText("AIからのフィードバックレポートを5分間読み、2回目のグループディスカッションに備えてください。")
        self._update_reading_countdown()
        self.reading_timer.start(1000)  # 1秒ごとに更新
    
    def _update_reading_countdown(self):
        """読書時間カウントダウンを更新"""
        if self.reading_remaining_seconds > 0:
            minutes = self.reading_remaining_seconds // 60
            seconds = self.reading_remaining_seconds % 60
            self.reading_countdown_label.setText(f"{minutes:02d}:{seconds:02d}")
            self.reading_remaining_seconds -= 1
        else:
            # 読書時間終了
            self.reading_timer.stop()
            self.reading_active = False
            self.reading_countdown_label.hide()
            self.reading_message_label.hide()
            # 自動的に2回目GD開始確認画面へ遷移
            self.reading_timeout.emit()
    
    def set_feedback(self, feedback_dict):
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
            # 改行を適切に処理して読みやすくする
            import re
            # まず、既存の改行を保持しつつ、段落間の改行を統一
            formatted_feedback = exp_feedback.strip()
            formatted_feedback = re.sub(r'\n{3,}', '\n\n', formatted_feedback)
            
            # セクション見出し（## や ###）の前後に改行を追加
            formatted_feedback = re.sub(r'\n(##\s+)', r'\n\n\1', formatted_feedback)
            formatted_feedback = re.sub(r'(##\s+[^\n]+)\n(?!\n)', r'\1\n\n', formatted_feedback)
            
            # 「Good」「More」「Action」などのセクション見出し（###）の前後に改行を追加
            formatted_feedback = re.sub(r'\n(###\s+)', r'\n\n\1', formatted_feedback)
            formatted_feedback = re.sub(r'(###\s+[^\n]+)\n(?!\n)', r'\1\n\n', formatted_feedback)
            
            # 「Good:」「More:」「Action:」などのキーワードの前後に改行を追加
            # 最初のGoodの前には区切り線を追加せず、MoreとActionの前には区切り線を追加
            # セクション間にもっとスペースを追加するため、区切り線の前後に改行を追加
            formatted_feedback = re.sub(r'\n(Good):\s*', r'\n\n**\1:**\n\n', formatted_feedback, flags=re.IGNORECASE)
            formatted_feedback = re.sub(r'\n(More|Action):\s*', r'\n\n\n---\n\n\n**\1:**\n\n', formatted_feedback, flags=re.IGNORECASE)
            
            # リスト項目（- や * で始まる行）の前に改行を追加（ただし連続するリスト項目の間は改行しない）
            formatted_feedback = re.sub(r'\n([-*]\s+)', r'\n\n\1', formatted_feedback)
            
            # 文の区切り（。や！や？の後）で改行を追加（ただし、既に改行がある場合は追加しない）
            formatted_feedback = re.sub(r'([。！？])\s+([^\n。！？\n])', r'\1\n\n\2', formatted_feedback)
            
            # 連続する改行を整理（セクション間の区切り線周辺は保持、それ以外は2つに統一）
            # 区切り線周辺の改行パターンを一時的に保護
            formatted_feedback = formatted_feedback.replace('\n\n\n---\n\n\n', '___SEPARATOR___')
            # 5つ以上の改行を4つに、3つ以上の改行を2つに統一
            formatted_feedback = re.sub(r'\n{5,}', '\n\n\n\n', formatted_feedback)
            formatted_feedback = re.sub(r'\n{3,}', '\n\n', formatted_feedback)
            # 区切り線を元に戻す（前後に3つの改行を保持）
            formatted_feedback = formatted_feedback.replace('___SEPARATOR___', '\n\n\n---\n\n\n')
            
            feedback_md += formatted_feedback + "\n\n"
        
        # Markdownとしてレンダリング
        self.feedback_text.setMarkdown(feedback_md)
    
    def show_progress(self, message):
        """進捗メッセージを表示"""
        self.progress_label.setText(message)
        self.progress_label.show()
        # フィードバックテキストをクリア
        self.feedback_text.setMarkdown("")


class MicrophoneCheckScreen(QWidget):
    """マイクチェック画面"""
    microphone_check_completed = pyqtSignal()  # マイクチェック完了シグナル
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 全画面表示に対応したレイアウト
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 中央コンテナ（最大幅を制限して全画面でも見やすく）
        container = QWidget()
        container.setMaximumWidth(1400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(40)
        container_layout.setContentsMargins(100, 80, 100, 80)
        
        container_layout.addStretch()
        
        # タイトル
        title_label = QLabel("マイクチェック")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 40px;
                font-weight: bold;
                color: #2c3e50;
                padding: 30px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title_label)
        
        # 説明文
        description_label = QLabel("マイクの動作を確認してください")
        description_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #34495e;
                padding: 15px;
            }
        """)
        description_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(description_label)
        
        container_layout.addSpacing(30)
        
        # 指定文言表示エリア（大きなフレーム）
        phrase_frame = QFrame()
        phrase_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 3px solid #3498db;
                border-radius: 10px;
                padding: 60px;
            }
        """)
        phrase_frame.setMaximumWidth(1200)  # 最大幅を制限して全画面でも見やすく
        phrase_frame_layout = QVBoxLayout()
        phrase_frame_layout.setSpacing(40)
        phrase_frame_layout.setContentsMargins(40, 40, 40, 40)
        
        # 指定文言表示（初期状態でも表示）
        self.phrase_label = QLabel("「音声テストを開始」ボタンを押してください")
        self.phrase_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                padding: 40px;
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                min-height: 120px;
            }
        """)
        self.phrase_label.setAlignment(Qt.AlignCenter)
        self.phrase_label.setWordWrap(True)
        phrase_frame_layout.addWidget(self.phrase_label)
        
        # リアルタイム音量表示
        volume_layout = QVBoxLayout()
        volume_layout.setSpacing(10)
        volume_label = QLabel("音量レベル")
        volume_label.setStyleSheet("font-size: 18px; color: #34495e; font-weight: bold;")
        volume_label.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(volume_label)
        
        self.volume_progress = QProgressBar()
        self.volume_progress.setMinimum(0)
        self.volume_progress.setMaximum(100)
        self.volume_progress.setValue(0)
        self.volume_progress.setFixedHeight(60)
        self.volume_progress.setStyleSheet("""
            QProgressBar {
                border: 3px solid #bdc3c7;
                border-radius: 10px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 7px;
            }
        """)
        self.volume_progress.setFormat("%p%")
        volume_layout.addWidget(self.volume_progress)
        phrase_frame_layout.addLayout(volume_layout)
        
        phrase_frame.setLayout(phrase_frame_layout)
        container_layout.addWidget(phrase_frame)
        
        container_layout.addSpacing(30)
        
        # 音声チェックボタン
        self.audio_test_button = QPushButton("音声テストを開始")
        self.audio_test_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 25px 80px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
                min-width: 300px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.audio_test_button.clicked.connect(self._start_audio_test)
        container_layout.addWidget(self.audio_test_button, alignment=Qt.AlignCenter)
        
        container_layout.addSpacing(20)
        
        # ステータス表示
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #7f8c8d;
                padding: 20px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        container_layout.addWidget(self.status_label)
        
        container_layout.addSpacing(30)
        
        # 次へボタン（最初は無効）
        self.next_button = QPushButton("次へ")
        self.next_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 25px 80px;
                background-color: #e67e22;
                color: white;
                border: none;
                border-radius: 10px;
                min-width: 300px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.microphone_check_completed.emit)
        container_layout.addWidget(self.next_button, alignment=Qt.AlignCenter)
        
        container_layout.addStretch()
        container.setLayout(container_layout)
        
        # 中央揃え
        main_layout.addStretch()
        main_layout.addWidget(container, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
        self.audio_checked = False
        self.recording = False
        
        # 指定文言（テスト用）
        self.test_phrase = "これは音声テストです"
        
        # タイマー
        self.volume_update_timer = QTimer()
        self.volume_update_timer.timeout.connect(self._update_volume_display)
    
    def _start_audio_test(self):
        """音声テストを開始（リアルタイム音量表示 + 指定文言チェック）"""
        self.phrase_label.setText(f'「{self.test_phrase}」\n\nと話してください')
        self.phrase_label.show()
        self.status_label.setText("マイクに向かって指定の文言を話してください...")
        self.audio_test_button.setEnabled(False)
        self.recording = True
        
        # リアルタイム音量表示を開始
        self.volume_update_timer.start(50)  # 50msごとに更新
        
        # 音声テストを実行（非同期）
        from threading import Thread
        thread = Thread(target=self._record_and_check_audio_with_phrase, daemon=True)
        thread.start()
    
    def _update_volume_display(self):
        """リアルタイム音量表示を更新"""
        if not self.recording:
            return
        
        try:
            import pyaudio
            import numpy as np
            
            RATE = 16000
            CHUNK = int(RATE / 10)
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            
            if not hasattr(self, '_p_audio') or self._p_audio is None:
                self._p_audio = pyaudio.PyAudio()
                self._stream = self._p_audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                max_volume = np.max(np.abs(audio_data))
                
                # 音量を0-100%に正規化（32767が最大値）
                volume_percent = min(100, int((max_volume / 32767) * 100))
                
                # プログレスバーを更新
                self.volume_progress.setValue(volume_percent)
                
                # 閾値チェック（1000以上でOK）
                if max_volume > 1000:
                    self.volume_progress.setStyleSheet("""
                        QProgressBar {
                            border: 3px solid #bdc3c7;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 16px;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: #2ecc71;
                            border-radius: 7px;
                        }
                    """)
                else:
                    self.volume_progress.setStyleSheet("""
                        QProgressBar {
                            border: 3px solid #bdc3c7;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 16px;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: #e74c3c;
                            border-radius: 7px;
                        }
                    """)
            except:
                pass
        except Exception as e:
            print(f"音量表示更新エラー: {e}")
    
    def _record_and_check_audio_with_phrase(self):
        """指定文言を録音してチェック（音声認識使用）"""
        try:
            import pyaudio
            import numpy as np
            import wave
            import os
            from google.cloud import speech_v1p1beta1 as speech
            
            RATE = 16000
            CHUNK = int(RATE / 10)
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RECORD_SECONDS = 5  # 5秒間録音
            
            p_audio = pyaudio.PyAudio()
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not self.recording:
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p_audio.terminate()
            
            # 録音を停止
            self.recording = False
            self.volume_update_timer.stop()
            if hasattr(self, '_stream'):
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except:
                    pass
            if hasattr(self, '_p_audio'):
                try:
                    self._p_audio.terminate()
                    self._p_audio = None
                except:
                    pass
            
            # 音量レベルをチェック
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            max_volume = np.max(np.abs(audio_data))
            
            # 音声認識で指定文言をチェック
            audio_content = b''.join(frames)
            
            # 一時ファイルに保存
            temp_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio_data", "audio_check_temp.wav")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p_audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_content)
            wf.close()
            
            # 音声認識を実行
            speech_client = speech.SpeechClient()
            with open(temp_file, 'rb') as audio_file:
                content = audio_file.read()
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code='ja-JP'
            )
            audio = speech.RecognitionAudio(content=content)
            
            response = speech_client.recognize(config=config, audio=audio)
            
            # 認識結果をチェック
            recognized_text = ""
            if response.results:
                recognized_text = response.results[0].alternatives[0].transcript
            
            # UIスレッドで更新
            from PySide6.QtCore import QTimer
            
            # 閾値チェックと文言チェック
            phrase_match = self.test_phrase in recognized_text or recognized_text in self.test_phrase
            volume_ok = max_volume > 1000
            
            if phrase_match and volume_ok:
                QTimer.singleShot(0, lambda: self._on_audio_check_success())
            elif not volume_ok:
                QTimer.singleShot(0, lambda: self._on_audio_check_failed("音量が低すぎます。もう少し大きな声で話してください。"))
            else:
                QTimer.singleShot(0, lambda: self._on_audio_check_failed(f"指定の文言が認識されませんでした。認識結果: {recognized_text}"))
            
            # 一時ファイルを削除
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            self.recording = False
            self.volume_update_timer.stop()
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._on_audio_check_error(str(e)))
    
    def _on_audio_check_success(self):
        """音声チェック成功"""
        self.audio_checked = True
        self.status_label.setText("✓ マイクチェック完了: マイクが正常に動作しています。")
        self.audio_test_button.setEnabled(True)
        self.phrase_label.setText("✓ チェック完了")
        self.volume_progress.setValue(0)
        self.volume_progress.setStyleSheet("""
            QProgressBar {
                border: 3px solid #bdc3c7;
                border-radius: 10px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #95a5a6;
                border-radius: 7px;
            }
        """)
        self.next_button.setEnabled(True)
    
    def _on_audio_check_failed(self, message="音声が検出されませんでした。マイクを確認してください。"):
        """音声チェック失敗"""
        self.status_label.setText(f"✗ {message}")
        self.audio_test_button.setEnabled(True)
        self.phrase_label.setText("「音声テストを開始」ボタンを押してください")
        self.volume_progress.setValue(0)
        self.volume_progress.setStyleSheet("""
            QProgressBar {
                border: 3px solid #bdc3c7;
                border-radius: 10px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #95a5a6;
                border-radius: 7px;
            }
        """)
    
    def _on_audio_check_error(self, error_msg):
        """音声チェックエラー"""
        self.recording = False
        self.volume_update_timer.stop()
        self.status_label.setText(f"エラー: {error_msg}")
        self.audio_test_button.setEnabled(True)
        if hasattr(self, '_stream'):
            try:
                self._stream.stop_stream()
                self._stream.close()
            except:
                pass
        if hasattr(self, '_p_audio'):
            try:
                self._p_audio.terminate()
                self._p_audio = None
            except:
                pass

class SpeakerCheckScreen(QWidget):
    """スピーカーチェック画面"""
    speaker_check_completed = pyqtSignal()  # スピーカーチェック完了シグナル
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 全画面表示に対応したレイアウト
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 中央コンテナ（最大幅を制限して全画面でも見やすく）
        container = QWidget()
        container.setMaximumWidth(900)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(25)
        container_layout.setContentsMargins(60, 40, 60, 40)
        
        container_layout.addStretch()
        
        # タイトル
        title_label = QLabel("スピーカーから音声が繰り返し再生されます。音量を調整してください。\n調整が終わったら「次へ」ボタンを押してください。")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                padding: 20px;
                line-height: 1.6;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(600)
        container_layout.addWidget(title_label)
        
        container_layout.addSpacing(30)
        
        # 音量チェックボタン（緑のボタン）
        self.volume_test_button = QPushButton("音量テストを開始")
        self.volume_test_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 30px;
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 5px;
                max-width: 400px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.volume_test_button.setMaximumWidth(400)
        self.volume_test_button.clicked.connect(self._start_volume_test)
        container_layout.addWidget(self.volume_test_button, alignment=Qt.AlignCenter)
        
        container_layout.addSpacing(20)
        
        # 次へボタン（緑のボタン）
        self.next_button = QPushButton("次へ")
        self.next_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 30px;
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 5px;
                max-width: 400px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        self.next_button.setMaximumWidth(400)
        self.next_button.setEnabled(False)
        container_layout.addWidget(self.next_button, alignment=Qt.AlignCenter)
        
        container_layout.addStretch()
        container.setLayout(container_layout)
        
        # 中央揃え
        main_layout.addStretch()
        main_layout.addWidget(container, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
        self.volume_checked = False
        self.volume_testing = False
        self.volume_test_thread = None
        self.volume_test_stop_flag = False
        self.current_audio_stream = None  # 現在再生中のオーディオストリーム
        self.current_p_audio = None  # 現在のPyAudioインスタンス
        self._audio_lock = threading.Lock()  # オーディオリソースへのアクセスを保護するロック
        
        # 次へボタンクリック時に音量テストを停止
        self.next_button.clicked.connect(self._stop_volume_test_on_next)
    
    def _start_volume_test(self):
        """音量テストを開始（ループ再生）"""
        self.volume_test_button.setEnabled(False)
        self.volume_testing = True
        self.volume_test_stop_flag = False
        
        # 音量テストを実行（非同期）
        from threading import Thread
        self.volume_test_thread = Thread(target=self._play_volume_test_loop, daemon=True)
        self.volume_test_thread.start()
        
        # 次へボタンを有効化（ユーザーが停止できるように）
        self.next_button.setEnabled(True)
    
    def _play_volume_test_loop(self):
        """テスト音声をループ再生（中断可能）"""
        try:
            from google.cloud import texttospeech_v1beta1 as texttospeech
            import pyaudio
            import numpy as np
            import time
            
            # 定数（gd_managerから取得）
            RATE = 24000
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            LANGUAGE_CODE_TTS = "ja-JP"
            
            test_message = "スピーカーから音声が再生されます。音量を調整してください。調整が終わったら「次へ」ボタンを押してください。"
            voice_name = "ja-JP-Neural2-D"
            
            tts_client = texttospeech.TextToSpeechClient()
            
            # 音声合成
            synthesis_input = texttospeech.SynthesisInput(text=test_message)
            voice = texttospeech.VoiceSelectionParams(
                language_code=LANGUAGE_CODE_TTS,
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                speaking_rate=1.2,
                pitch=0.0
            )
            
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content
            
            # ループ再生
            while self.volume_testing and not self.volume_test_stop_flag:
                if self.volume_test_stop_flag:
                    break
                
                try:
                    # ロックを取得してリソースを安全に作成
                    with self._audio_lock:
                        if self.volume_test_stop_flag:
                            break
                        self.current_p_audio = pyaudio.PyAudio()
                        stream = self.current_p_audio.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            output=True,
                            frames_per_buffer=CHUNK
                        )
                        self.current_audio_stream = stream
                    stream.start_stream()
                    
                    # 無音データを先頭に追加
                    silence_chunks = 3
                    silence_data = np.zeros(CHUNK * silence_chunks, dtype=np.int16)
                    audio_data = np.frombuffer(audio_content, dtype=np.int16)
                    combined_audio = np.concatenate([silence_data, audio_data])
                    total_frames = len(combined_audio)
                    
                    for i in range(0, total_frames, CHUNK):
                        if self.volume_test_stop_flag:
                            break
                        chunk_data = combined_audio[i:i+CHUNK]
                        if len(chunk_data) < CHUNK:
                            chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), mode='constant')
                        try:
                            stream.write(chunk_data.tobytes())
                        except:
                            # ストリームが既に閉じられている可能性
                            break
                    
                    # ロックを取得してリソースを安全にクリーンアップ
                    with self._audio_lock:
                        if not self.volume_test_stop_flag:
                            try:
                                stream.stop_stream()
                            except:
                                pass
                        try:
                            stream.close()
                        except:
                            pass
                        try:
                            if self.current_p_audio:
                                self.current_p_audio.terminate()
                        except:
                            pass
                        self.current_audio_stream = None
                        self.current_p_audio = None
                    
                    if self.volume_test_stop_flag:
                        break
                    
                    time.sleep(0.5)  # 少し待機してから次の再生
                    
                except Exception as e:
                    # エラーが発生してもリソースをクリーンアップ
                    with self._audio_lock:
                        if self.current_audio_stream:
                            try:
                                if hasattr(self.current_audio_stream, 'is_active'):
                                    try:
                                        if self.current_audio_stream.is_active():
                                            self.current_audio_stream.stop_stream()
                                    except:
                                        pass
                                self.current_audio_stream.close()
                            except:
                                pass
                            self.current_audio_stream = None
                        if self.current_p_audio:
                            try:
                                self.current_p_audio.terminate()
                            except:
                                pass
                            self.current_p_audio = None
                    # 停止フラグが立っている場合はエラーを無視して終了
                    if self.volume_test_stop_flag:
                        break
                    # 停止フラグが立っていない場合もエラーを無視して続行（無限ループを防ぐ）
                    print(f"[警告]: 音声再生中にエラー: {e}")
                    time.sleep(0.5)  # エラー後も少し待機
            
        except Exception as e:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._on_volume_check_error(str(e)))
    
    def _stop_volume_test_on_next(self):
        """次へボタンクリック時に音量テストを停止（音声も中断）"""
        try:
            # 停止フラグを設定（スレッドに停止を通知）
            self.volume_testing = False
            self.volume_test_stop_flag = True
            
            # ロックを取得してリソースを安全にクリーンアップ
            with self._audio_lock:
                # 現在再生中の音声を中断
                stream = self.current_audio_stream
                p_audio = self.current_p_audio
                
                if stream:
                    try:
                        # ストリームが有効かチェック
                        if hasattr(stream, 'is_active'):
                            try:
                                if stream.is_active():
                                    stream.stop_stream()
                            except:
                                pass  # 既に停止している可能性
                        stream.close()
                    except Exception as e:
                        print(f"[警告]: 音声ストリームの停止中にエラー: {e}")
                    finally:
                        self.current_audio_stream = None
                
                if p_audio:
                    try:
                        # すべてのストリームを閉じてからterminate
                        p_audio.terminate()
                    except Exception as e:
                        print(f"[警告]: PyAudioの終了中にエラー: {e}")
                    finally:
                        self.current_p_audio = None
            
            # スレッドの終了を少し待つ（最大0.5秒）
            if self.volume_test_thread and self.volume_test_thread.is_alive():
                self.volume_test_thread.join(timeout=0.5)
            
            self.volume_checked = True
            
            # 全画面表示にする
            window = self.window()
            if window:
                try:
                    window.showFullScreen()
                except Exception as e:
                    print(f"[警告]: 全画面表示の設定中にエラー: {e}")
            
            # シグナルを発火（例外が発生しても実行）
            try:
                self.speaker_check_completed.emit()
            except Exception as e:
                print(f"[警告]: シグナル発火中にエラー: {e}")
                
        except Exception as e:
            print(f"[エラー]: 音量テスト停止処理中にエラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもシグナルは発火する
            try:
                self.speaker_check_completed.emit()
            except:
                pass
    
    def _on_volume_check_error(self, error_msg):
        """音量チェックエラー"""
        self.volume_testing = False
        self.volume_test_stop_flag = True
        self.volume_test_button.setEnabled(True)
        print(f"[エラー]: {error_msg}")

class ControlGroupAfterFirstScreen(QWidget):
    """統制群用: 1回目終了後の画面（学習用ドキュメント表示）"""
    reading_timeout = pyqtSignal()  # 読書時間終了時に発火
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 読書時間カウントダウンラベル
        self.reading_countdown_label = QLabel()
        self.reading_countdown_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #e74c3c;
                margin-bottom: 10px;
            }
        """)
        self.reading_countdown_label.setAlignment(Qt.AlignCenter)
        self.reading_countdown_label.hide()  # 初期状態は非表示
        
        # 読書時間メッセージラベル
        self.reading_message_label = QLabel()
        self.reading_message_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        self.reading_message_label.setAlignment(Qt.AlignCenter)
        self.reading_message_label.hide()  # 初期状態は非表示
        
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
        
        layout.addWidget(self.reading_countdown_label)
        layout.addWidget(self.reading_message_label)
        layout.addWidget(self.doc_text)
        
        self.setLayout(layout)
        
        # 読書時間タイマー
        self.reading_timer = QTimer()
        self.reading_timer.timeout.connect(self._update_reading_countdown)
        self.reading_remaining_seconds = 0
        self.reading_active = False
    
    def start_reading_time(self, seconds=300):
        """読書時間を開始（5分間、開発モードの場合は10秒）"""
        self.reading_remaining_seconds = seconds
        self.reading_active = True
        self.reading_countdown_label.show()
        self.reading_message_label.show()
        # 時間に応じてメッセージを変更（アナウンス文言に合わせる）
        if seconds <= 30:
            self.reading_message_label.setText("ファシリテーションに関するハンドブックを10秒間読み、2回目のグループディスカッションに備えてください。")
        else:
            self.reading_message_label.setText("ファシリテーションに関するハンドブックを5分間読み、2回目のグループディスカッションに備えてください。")
        self._update_reading_countdown()
        self.reading_timer.start(1000)  # 1秒ごとに更新
    
    def _update_reading_countdown(self):
        """読書時間カウントダウンを更新"""
        if self.reading_remaining_seconds > 0:
            minutes = self.reading_remaining_seconds // 60
            seconds = self.reading_remaining_seconds % 60
            self.reading_countdown_label.setText(f"{minutes:02d}:{seconds:02d}")
            self.reading_remaining_seconds -= 1
        else:
            # 読書時間終了
            self.reading_timer.stop()
            self.reading_active = False
            self.reading_countdown_label.hide()
            self.reading_message_label.hide()
            # 自動的に2回目GD開始確認画面へ遷移
            self.reading_timeout.emit()
