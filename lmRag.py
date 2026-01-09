import sys
import json
import requests
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QComboBox, QLabel, QTabWidget, QListWidget, 
                             QSplitter, QMessageBox, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QFont, QIcon
import chromadb
from chromadb.config import Settings
import uuid

# Markdown desteği için deneyelim
try:
    import markdown
    HAVE_MARKDOWN = True
except ImportError:
    HAVE_MARKDOWN = False

class ChatThread(QThread):
    response_received = pyqtSignal(str)
    response_chunk = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, url, model, messages, use_rag=False, rag_context=""):
        super().__init__()
        self.url = url
        self.model = model
        self.messages = messages
        self.use_rag = use_rag
        self.rag_context = rag_context
        self._is_running = True

    def stop(self):
        """Dışarıdan çağrılarak döngüyü durdurur."""
        self._is_running = False

    def run(self):
        try:
            self._is_running = True
            if self.use_rag and self.rag_context:
                enhanced_messages = self.messages.copy()
                enhanced_messages[-1]["content"] = f"İlgili bilgiler:\n{self.rag_context}\n\nSoru: {self.messages[-1]['content']}"
                messages_to_send = enhanced_messages
            else:
                messages_to_send = self.messages

            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages_to_send,
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    # Durdurma kontrolü
                    if not self._is_running:
                        break
                        
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]
                            if line.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(line)
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        self.response_chunk.emit(content)
                            except json.JSONDecodeError:
                                continue
                
                # Eğer durdurulmadıysa tamamlanma sinyali gönder
                if self._is_running:
                    self.response_received.emit(full_response)
            else:
                self.error_occurred.emit(f"Hata: {response.status_code}")
        except Exception as e:
            self.error_occurred.emit(f"Bağlantı hatası: {str(e)}")

class LMStudioRAGChat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LM Studio RAG Chat")
        self.setGeometry(100, 100, 1300, 800)
        
        # Klasör tanımları
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.chat_history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_histories")
        os.makedirs(self.chat_history_dir, exist_ok=True)
        
        self.current_chat_id = None
        self.is_new_chat = True
        self.expecting_completion = False # Durdurma butonu için bayrak
        
        self.setup_dark_theme()
        
        # ChromaDB Ayarları
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            persist_directory=self.data_dir,
            is_persistent=True
        ))
        
        try:
            self.collection = self.chroma_client.get_collection("rag_knowledge")
        except:
            self.collection = self.chroma_client.create_collection("rag_knowledge")
        
        self.lm_studio_url = "http://localhost:1234"
        self.current_model = ""
        self.chat_history = []
        self.current_response = ""
        
        self.setup_ui()
        self.load_models()
        self.load_chat_list()
        
        # Başlangıçta yeni bir sohbet oluştur
        self.new_chat()

    def format_response(self, text):
        """Metni HTML'e formatlar (Markdown, kod blokları, tablolar)"""
        # Markdown'dan HTML'e dönüştür
        if HAVE_MARKDOWN:
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            # Kod blokları için stil ekle
            def add_style_to_code(match):
                code_block = match.group(0)
                if '<pre><code>' in code_block or '<pre><code ' in code_block:
                    code_block = re.sub(
                        r'<pre>',
                        r'<pre style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; overflow-x: auto; margin: 10px 0;">',
                        code_block
                    )
                return code_block
            html = re.sub(r'<pre>.*?</pre>', add_style_to_code, html, flags=re.DOTALL)
            # Tablolar için stil ekle
            html = html.replace('<table>', '<table style="border-collapse: collapse; width: 100%; margin: 15px 0;">')
            html = html.replace('<th>', '<th style="border: 1px solid #30363d; padding: 8px; background-color: #0d1117;">')
            html = html.replace('<td>', '<td style="border: 1px solid #30363d; padding: 8px;">')
            return html
        else:
            # Basit dönüşüm - kod blokları ve satır içi kod
            # Kod blokları
            def replace_code_block(match):
                lang = match.group(1) or ''
                code = match.group(2)
                return f'<pre style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; overflow-x: auto; margin: 10px 0;"><code class="language-{lang}">{code}</code></pre>'
            
            text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
            
            # Satır içi kod
            text = re.sub(
                r'`([^`]+)`',
                r'<code style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 3px; padding: 2px 6px; font-family: \'Courier New\', monospace;">\1</code>',
                text
            )
            
            # Satır sonlarını <br> ile değiştir
            text = text.replace('\n', '<br>')
            
            # Basit tablo desteği (Markdown tabloları)
            lines = text.split('<br>')
            in_table = False
            table_html = ''
            for i, line in enumerate(lines):
                if '|' in line and ('---' in line or i > 0 and '|' in lines[i-1]):
                    if not in_table:
                        in_table = True
                        table_html = '<table style="border-collapse: collapse; width: 100%; margin: 15px 0;">'
                    
                    # Tablo satırı
                    cells = [cell.strip() for cell in line.split('|') if cell.strip() != '']
                    if '---' in line:
                        # Header ayırıcı - bu satırı atla
                        continue
                    
                    table_html += '<tr>'
                    for cell in cells:
                        if i == 0 or (i > 0 and '---' in lines[i-1]):
                            # Header hücresi
                            table_html += f'<th style="border: 1px solid #30363d; padding: 8px; background-color: #0d1117;">{cell}</th>'
                        else:
                            # Normal hücre
                            table_html += f'<td style="border: 1px solid #30363d; padding: 8px;">{cell}</td>'
                    table_html += '</tr>'
                else:
                    if in_table:
                        in_table = False
                        table_html += '</table>'
                        text = text.replace(line, table_html + line)
            
            return text

    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e6edf3;
            }
            QTextEdit {
                background-color: #0d1117;
                color: #e6edf3;
                border: none;
                padding: 20px;
                font-size: 15px;
                font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.7;
            }
            QLineEdit {
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px 15px;
                font-size: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLineEdit:focus {
                border: 1px solid #58a6ff;
                background-color: #0d1117;
            }
            QListWidget {
                background-color: #0d1117;
                color: #e6edf3;
                border: none;
                outline: none;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #21262d;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #161b22;
            }
            QListWidget::item:selected {
                background-color: #1f6feb;
                color: white;
            }
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:pressed {
                background-color: #1a7f37;
            }
            QPushButton:disabled {
                background-color: #21262d;
                color: #6e7681;
            }
            QComboBox {
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e6edf3;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background-color: #0d1117;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #161b22;
                color: #8b949e;
                padding: 10px 24px;
                border: 1px solid #30363d;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0d1117;
                color: #e6edf3;
                border-bottom: 2px solid #58a6ff;
            }
            QTabBar::tab:hover {
                background-color: #1c2128;
                color: #e6edf3;
            }
            QLabel {
                color: #e6edf3;
                font-size: 13px;
            }
        """)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Ana Yatay Düzen (Sol Panel + Sağ İçerik)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- SOL PANEL (Sohbet Geçmişi) ---
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #161b22; border-right: 1px solid #30363d;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Yeni Sohbet Butonu
        self.new_chat_btn = QPushButton("+ Yeni Sohbet")
        self.new_chat_btn.clicked.connect(self.new_chat)
        self.new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                padding: 12px;
                text-align: left;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
        """)
        sidebar_layout.addWidget(self.new_chat_btn)
        
        sidebar_layout.addWidget(QLabel("Geçmiş Sohbetler:"))
        
        # Sohbet Listesi
        self.chat_list = QListWidget()
        self.chat_list.itemClicked.connect(self.load_chat)
        sidebar_layout.addWidget(self.chat_list)

        # Sohbet Silme Butonu (Yeni Eklendi)
        self.delete_chat_btn = QPushButton("🗑️ Seçili Sohbeti Sil")
        self.delete_chat_btn.clicked.connect(self.delete_selected_chat)
        self.delete_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #da3633;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #f85149;
            }
        """)
        sidebar_layout.addWidget(self.delete_chat_btn)
        
        # --- SAĞ PANEL (Ana İçerik) ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Üst Kısım (Model Seçimi ve Sohbeti Temizle)
        top_layout = QHBoxLayout()
        
        # Sol taraf - Model işlemleri
        top_left_layout = QVBoxLayout()
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_row.addWidget(self.model_combo)
        
        self.refresh_btn = QPushButton("🔄 Yenile")
        self.refresh_btn.clicked.connect(self.load_models)
        model_row.addWidget(self.refresh_btn)
        top_left_layout.addLayout(model_row)

        # Sağ taraf - Sohbet işlemleri
        top_right_layout = QHBoxLayout()
        top_right_layout.addStretch()
        
        self.clear_btn = QPushButton("🧹 Sohbet Ekranını Temizle")
        self.clear_btn.clicked.connect(self.clear_current_chat)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #8b949e;
                border: 1px solid #30363d;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #e6edf3;
            }
        """)
        top_right_layout.addWidget(self.clear_btn)

        top_layout.addLayout(top_left_layout)
        top_layout.addLayout(top_right_layout)
        
        content_layout.addLayout(top_layout)
        
        # Sekmeler
        self.tabs = QTabWidget()
        
        chat_tab = self.create_chat_tab()
        self.tabs.addTab(chat_tab, "💬 Sohbet")
        
        rag_tab = self.create_rag_tab()
        self.tabs.addTab(rag_tab, "📚 RAG Yönetimi")
        
        content_layout.addWidget(self.tabs)
        
        # Panelleri Ana Düzene Ekle
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_widget)

    def create_chat_tab(self):
        chat_widget = QWidget()
        layout = QVBoxLayout(chat_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        # Chat display'e CSS ekle
        self.chat_display.setHtml("""
        <style>
            pre {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px;
                overflow-x: auto;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 14px;
            }
            code {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 3px;
                padding: 2px 6px;
                font-family: 'Courier New', monospace;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }
            th, td {
                border: 1px solid #30363d;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #0d1117;
            }
            .user-message {
                margin: 15px 40px 8px 120px;
                animation: slideInRight 0.3s ease-out;
            }
            .assistant-message {
                margin: 8px 120px 15px 40px;
                animation: slideInLeft 0.3s ease-out;
            }
            .message-divider {
                margin: 8px 80px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
        </style>
        """)
        layout.addWidget(self.chat_display)
        
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                border-top: 1px solid #30363d;
                padding: 15px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        
        message_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Mesajınızı yazın...")
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setMinimumHeight(45)
        message_layout.addWidget(self.message_input)
        
        self.send_btn = QPushButton("Gönder →")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setMinimumWidth(100)
        message_layout.addWidget(self.send_btn)

        # Durdur Butonu
        self.stop_btn = QPushButton("🛑 Durdur")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #9e6a03;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d29922;
            }
        """)
        message_layout.addWidget(self.stop_btn)
        
        input_layout.addLayout(message_layout)
        layout.addWidget(input_container)
        
        return chat_widget

    def create_rag_tab(self):
        rag_widget = QWidget()
        layout = QVBoxLayout(rag_widget)
        
        layout.addWidget(QLabel("📝 Yeni Bilgi Ekle:"))
        self.rag_input = QTextEdit()
        self.rag_input.setPlaceholderText("Öğretmek istediğiniz bilgileri buraya yazın...")
        self.rag_input.setMaximumHeight(200)
        layout.addWidget(self.rag_input)
        
        rag_btn_layout = QHBoxLayout()
        self.add_rag_btn = QPushButton("✅ RAG'a Ekle")
        self.add_rag_btn.clicked.connect(self.add_to_rag)
        rag_btn_layout.addWidget(self.add_rag_btn)
        rag_btn_layout.addStretch()
        layout.addLayout(rag_btn_layout)
        
        layout.addWidget(QLabel("📚 Kayıtlı Bilgiler:"))
        self.rag_list = QListWidget()
        layout.addWidget(self.rag_list)
        
        delete_layout = QHBoxLayout()
        self.delete_rag_btn = QPushButton("🗑️ Seçili Bilgiyi Sil")
        self.delete_rag_btn.clicked.connect(self.delete_rag)
        delete_layout.addWidget(self.delete_rag_btn)
        
        self.clear_all_rag_btn = QPushButton("⚠️ Tüm RAG'ı Temizle")
        self.clear_all_rag_btn.clicked.connect(self.clear_all_rag)
        delete_layout.addWidget(self.clear_all_rag_btn)
        
        delete_layout.addStretch()
        layout.addLayout(delete_layout)
        
        self.load_rag_list()
        
        return rag_widget

    # --- SOHBET YÖNETİM FONKSİYONLARI ---

    def get_chat_file_path(self, chat_id):
        return os.path.join(self.chat_history_dir, f"{chat_id}.json")

    def save_chat(self):
        if not self.current_chat_id:
            return
        
        data = {
            "id": self.current_chat_id,
            "title": self.get_current_chat_title(),
            "messages": self.chat_history,
            "html": self.chat_display.toHtml(),
            "timestamp": str(uuid.uuid4())
        }
        
        try:
            with open(self.get_chat_file_path(self.current_chat_id), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            current_item = self.chat_list.currentItem()
            if current_item and current_item.data(Qt.ItemDataRole.UserRole) == self.current_chat_id:
                current_item.setText(data["title"])
        except Exception as e:
            print(f"Kayıt hatası: {e}")

    def get_current_chat_title(self):
        if not self.chat_history:
            return "Yeni Sohbet"
        for msg in self.chat_history:
            if msg["role"] == "user":
                title = msg["content"].replace('\n', ' ')
                return (title[:30] + '...') if len(title) > 30 else title
        return "Yeni Sohbet"

    def load_chat_list(self):
        self.chat_list.clear()
        files = sorted([f for f in os.listdir(self.chat_history_dir) if f.endswith('.json')], reverse=True)
        
        for file in files:
            try:
                with open(os.path.join(self.chat_history_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    item = QListWidgetItem(data.get("title", "Adsız Sohbet"))
                    item.setData(Qt.ItemDataRole.UserRole, data["id"])
                    self.chat_list.addItem(item)
            except Exception as e:
                print(f"Dosya okuma hatası {file}: {e}")

    def new_chat(self):
        if not self.is_new_chat:
            self.save_chat()
            
        self.current_chat_id = str(uuid.uuid4())
        self.chat_history = []
        self.chat_display.clear()
        self.message_input.clear()
        self.is_new_chat = True
        self.chat_list.clearSelection()

    def load_chat(self, item):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        
        if self.current_chat_id and self.current_chat_id != chat_id:
            self.save_chat()
            
        file_path = self.get_chat_file_path(chat_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.current_chat_id = data["id"]
                self.chat_history = data["messages"]
                self.chat_display.setHtml(data["html"])
                self.is_new_chat = False
                
                cursor = self.chat_display.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self.chat_display.setTextCursor(cursor)
                
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Sohbet yüklenirken hata oluştu: {e}")

    def clear_current_chat(self):
        """Sadece görüntüyü temizler, geçmiş dosyasını silmez."""
        if not self.is_new_chat:
            reply = QMessageBox.question(self, "Onay", 
                                         "Mevcut sohbetin içeriğini ekrandan temizlemek istiyor musunuz? (Dosya silinmeyecek)",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.chat_display.clear()
        else:
            self.chat_display.clear()

    def delete_selected_chat(self):
        """Seçili sohbeti diskten ve listeden siler."""
        current_item = self.chat_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Uyarı", "Lütfen silmek için bir sohbet seçin.")
            return
        
        chat_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(self, "Onay", 
                                     "Bu sohbeti tamamen silmek istiyor musunuz? Bu işlem geri alınamaz.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Dosyayı sil
                file_path = self.get_chat_file_path(chat_id)
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Listeden sil
                self.chat_list.takeItem(self.chat_list.row(current_item))
                
                # Eğer silinen sohbet şu an açıksa, yeni sohbete geç
                if self.current_chat_id == chat_id:
                    self.new_chat()
                    
                QMessageBox.information(self, "Başarılı", "Sohbet silindi.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Sohbet silinirken hata oluştu: {str(e)}")

    # --- MEVCUT FONKSİYONLAR ---

    def load_models(self):
        try:
            response = requests.get(f"{self.lm_studio_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.model_combo.clear()
                for model in models['data']:
                    self.model_combo.addItem(model['id'])
                if self.model_combo.count() > 0:
                    self.current_model = self.model_combo.currentText()
        except Exception as e:
            QMessageBox.warning(self, "Bağlantı Hatası", 
                              f"LM Studio'ya bağlanılamadı.\nURL: {self.lm_studio_url}\nHata: {str(e)}")

    def on_model_changed(self, model_name):
        self.current_model = model_name

    def send_message(self):
        message = self.message_input.text().strip()
        if not message or not self.current_model:
            return
        
        if self.is_new_chat:
            self.is_new_chat = False
            title = (message[:30] + '...') if len(message) > 30 else message
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, self.current_chat_id)
            self.chat_list.insertItem(0, item)
            self.chat_list.setCurrentItem(item)
        
        self.message_input.clear()
        
        # Buton durumlarını güncelle: Gönder kapat, Durdur aç
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.message_input.setEnabled(False)
        self.expecting_completion = True
        
        # Kullanıcı mesajı
        escaped_message = message.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        self.chat_display.append(f"""
        <div class="user-message">
            <div style='display: flex; justify-content: flex-end; align-items: flex-start;'>
                <div style='max-width: 75%; background: linear-gradient(135deg, #1f6feb 0%, #388bfd 50%, #58a6ff 100%); 
                            padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
                            box-shadow: 0 4px 12px rgba(31, 111, 235, 0.25), 0 2px 4px rgba(31, 111, 235, 0.15);
                            position: relative;
                            backdrop-filter: blur(10px);
                            border: 1px solid rgba(88, 166, 255, 0.3);'>
                    <div style='color: #ffffff; font-size: 15px; line-height: 1.7; word-wrap: break-word; 
                                font-family: "Segoe UI", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif; 
                                font-weight: 500; letter-spacing: 0.3px;'>
                        {escaped_message}
                    </div>
                    <div style='text-align: right; margin-top: 6px; opacity: 0.75;'>
                        <span style='font-size: 11px; color: rgba(255, 255, 255, 0.8);'>Siz</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="message-divider">
            <div style='flex: 1; height: 1px; background: linear-gradient(90deg, transparent 0%, #30363d 20%, #30363d 80%, transparent 100%);'></div>
            <div style='padding: 4px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 12px;'>
                <span style='font-size: 10px; color: #8b949e; letter-spacing: 1px; font-weight: 600;'>YANIT</span>
            </div>
            <div style='flex: 1; height: 1px; background: linear-gradient(90deg, transparent 0%, #30363d 20%, #30363d 80%, transparent 100%);'></div>
        </div>
        """)
        self.chat_history.append({"role": "user", "content": message})
        
        rag_context = self.search_rag(message)
        
        self.chat_display.append("""
        <div class="assistant-message">
            <div style='display: flex; align-items: flex-start;'>
                <div style='width: 40px; height: 40px; 
                            background: linear-gradient(135deg, #238636 0%, #2ea043 50%, #3fb950 100%); 
                            border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                            margin-right: 12px; flex-shrink: 0; 
                            box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3), 0 2px 4px rgba(35, 134, 54, 0.2);
                            border: 2px solid rgba(63, 185, 80, 0.4);'>
                    <span style='font-size: 20px;'>🤖</span>
                </div>
                <div style='flex: 1; max-width: calc(100% - 52px);'>
                    <div style='background: linear-gradient(135deg, #161b22 0%, #1c2128 100%); 
                                padding: 14px 18px; border-radius: 18px 18px 18px 4px; 
                                border: 1px solid #30363d; 
                                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
                                position: relative;'>
                        <div style='color: #e6edf3; font-size: 15px; line-height: 1.8; word-wrap: break-word; 
                                    font-family: "Segoe UI", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif; 
                                    font-weight: 400; letter-spacing: 0.2px;'>""")
        
        self.current_response = ""
        self.response_start_pos = self.chat_display.textCursor().position()
        
        self.chat_thread = ChatThread(
            self.lm_studio_url, 
            self.current_model, 
            self.chat_history,
            use_rag=bool(rag_context),
            rag_context=rag_context
        )
        self.chat_thread.response_chunk.connect(self.on_response_chunk)
        self.chat_thread.response_received.connect(self.on_response_complete)
        self.chat_thread.error_occurred.connect(self.on_error)
        self.chat_thread.start()

    def on_response_chunk(self, chunk):
        self.current_response += chunk
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_response_complete(self, full_response):
        # Eğer durdurma butonuna basıldıysa ve bu fonksiyon sonradan çağrılıyorsa engelle
        if not self.expecting_completion:
            return

        # Akış sırasında eklenen metni seç ve sil
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self.response_start_pos)
        # Seçimi yap: self.current_response uzunluğu kadar karakter
        cursor.movePosition(QTextCursor.MoveOperation.Right, 
                            QTextCursor.MoveMode.KeepAnchor, 
                            len(self.current_response))
        # Seçili metni sil
        cursor.removeSelectedText()

        # Formatlanmış metni ekle
        formatted_html = self.format_response(full_response)
        cursor.insertHtml(formatted_html)
        
        # Kapanış HTML'ini ekle
        cursor.insertHtml("""
                        </div>
                        <div style='text-align: left; margin-top: 8px; padding-top: 8px; 
                                    border-top: 1px solid rgba(48, 54, 61, 0.5);'>
                            <span style='font-size: 11px; color: #8b949e; opacity: 0.8;'>
                                ✓ Yanıt tamamlandı
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <br>
        """)
        
        self.chat_history.append({"role": "assistant", "content": full_response})
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.message_input.setEnabled(True)
        self.current_response = ""
        self.expecting_completion = False
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        
        self.save_chat()

    def stop_generation(self):
        if hasattr(self, 'chat_thread') and self.chat_thread.isRunning():
            self.chat_thread.stop()
            self.expecting_completion = False
            
            # UI Durumu güncelle
            self.send_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.message_input.setEnabled(True)
            self.message_input.setFocus()
            
            # Akış sırasında eklenen metni seç ve sil, formatlanmış halini ekle
            if self.current_response:
                cursor = self.chat_display.textCursor()
                cursor.setPosition(self.response_start_pos)
                cursor.movePosition(QTextCursor.MoveOperation.Right, 
                                    QTextCursor.MoveMode.KeepAnchor, 
                                    len(self.current_response))
                cursor.removeSelectedText()
                formatted_html = self.format_response(self.current_response)
                cursor.insertHtml(formatted_html)
            
            # Görsel olarak durdurulduğunu belirt
            cursor.insertHtml("""
                        </div>
                        <div style='text-align: left; margin-top: 8px; padding-top: 8px; 
                                    border-top: 1px solid rgba(218, 54, 51, 0.5);'>
                            <span style='font-size: 11px; color: #f85149; opacity: 0.9;'>
                                ⚠️ Yanıt durduruldu
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            <br>
            """)
            self.chat_display.setTextCursor(cursor)
            
            if self.current_response:
                self.chat_history.append({"role": "assistant", "content": self.current_response})
                self.save_chat()
                self.current_response = ""

    def on_error(self, error_msg):
        self.expecting_completion = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.message_input.setEnabled(True)

        escaped_error = error_msg.replace('<', '&lt;').replace('>', '&gt;')
        self.chat_display.append(f"""
        <div style='margin: 20px 60px; text-align: center;'>
            <div style='display: inline-block; 
                        background: linear-gradient(135deg, #da3633 0%, #f85149 100%); 
                        padding: 12px 20px; border-radius: 20px; 
                        box-shadow: 0 4px 12px rgba(248, 81, 73, 0.3), 0 2px 4px rgba(248, 81, 73, 0.2);
                        border: 1px solid rgba(255, 255, 255, 0.15);
                        animation: shake 0.5s ease-in-out;'>
                <span style='color: #ffffff; font-size: 14px; font-weight: 500;
                            font-family: "Segoe UI", Arial, sans-serif;'>
                    ⚠️ {escaped_error}
                </span>
            </div>
        </div>
        <br>
        """)

    def add_to_rag(self):
        content = self.rag_input.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Uyarı", "Lütfen eklemek için bir bilgi yazın.")
            return
        
        try:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[content],
                ids=[doc_id],
                metadatas=[{"timestamp": str(uuid.uuid4())}]
            )
            self.rag_input.clear()
            self.load_rag_list()
            QMessageBox.information(self, "Başarılı", "Bilgi RAG'a eklendi!")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"RAG'a eklenirken hata: {str(e)}")

    def search_rag(self, query):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            if results['documents'] and results['documents'][0]:
                return "\n\n".join(results['documents'][0])
            return ""
        except Exception as e:
            print(f"RAG arama hatası: {e}")
            return ""

    def load_rag_list(self):
        self.rag_list.clear()
        try:
            all_docs = self.collection.get()
            for doc_id, content in zip(all_docs['ids'], all_docs['documents']):
                preview = content[:100] + "..." if len(content) > 100 else content
                self.rag_list.addItem(f"[{doc_id[:8]}] {preview}")
        except Exception as e:
            print(f"RAG yükleme hatası: {e}")

    def delete_rag(self):
        current_item = self.rag_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Uyarı", "Lütfen silmek için bir bilgi seçin.")
            return
        
        text = current_item.text()
        doc_id_short = text.split("]")[0].strip("[")
        
        try:
            all_docs = self.collection.get()
            for full_id in all_docs['ids']:
                if full_id.startswith(doc_id_short):
                    self.collection.delete(ids=[full_id])
                    self.load_rag_list()
                    QMessageBox.information(self, "Başarılı", "Bilgi silindi!")
                    return
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Silme hatası: {str(e)}")

    def clear_all_rag(self):
        reply = QMessageBox.question(self, "Onay", 
                                     "Tüm RAG bilgilerini silmek istediğinizden emin misiniz?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.chroma_client.delete_collection("rag_knowledge")
                self.collection = self.chroma_client.create_collection("rag_knowledge")
                self.load_rag_list()
                QMessageBox.information(self, "Başarılı", "Tüm RAG bilgileri temizlendi!")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Temizleme hatası: {str(e)}")
    
    def closeEvent(self, event):
        self.save_chat()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LMStudioRAGChat()
    window.show()
    sys.exit(app.exec())