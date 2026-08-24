import sys
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QLabel, 
                             QTextEdit, QFileDialog, QSplitter, QListWidgetItem,
                             QGridLayout, QFrame)
from PyQt6.QtCore import Qt, QProcess, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon

def ansi_to_html(text):
    """Parses ANSI terminal color codes and converts them to HTML for QTextEdit."""
    # Basic ANSI colors mapping to hex
    colors = {
        '30': '#000000', '31': '#ff5555', '32': '#50fa7b', '33': '#f1fa8c',
        '34': '#bd93f9', '35': '#ff79c6', '36': '#8be9fd', '37': '#f8f8f2',
        '90': '#6272a4', '91': '#ff6e6e', '92': '#69ff94', '93': '#ffffa5',
        '94': '#d6acff', '95': '#ff92df', '96': '#a4ffff', '97': '#ffffff'
    }
    
    html = text.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    # Regex to find ANSI escape sequences
    ansi_escape = re.compile(r'\x1b\[([\d;]+)m')
    
    parts = ansi_escape.split(html)
    result = ""
    open_span = False
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result += part
        else:
            codes = part.split(';')
            color = None
            for code in codes:
                if code in colors:
                    color = colors[code]
                elif code == '0':
                    if open_span:
                        result += "</span>"
                        open_span = False
            
            if color:
                if open_span:
                    result += "</span>"
                result += f"<span style='color: {color};'>"
                open_span = True
                
    if open_span:
        result += "</span>"
        
    return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helios Image Registration Pipeline")
        self.resize(1100, 800)
        self.setup_ui()
        self.process = None
        
        # Buffer for stdout to handle split lines when regex matching
        self.stdout_buffer = ""
        
        # Regex patterns for metrics extraction (strips ANSI internally first)
        self.metric_patterns = {
            "RMSE": re.compile(r'RMSE\s*\(Accuracy\)[^\d]*([\d.]+\s*px)'),
            "PSNR": re.compile(r'PSNR\s*\(Radiometric\)[^\d]*([\d.]+\s*dB)'),
            "SSIM": re.compile(r'SSIM\s*\(Structural\)[^\d]*([\d.]+)'),
            "Inlier Ratio": re.compile(r'Inlier Ratio[^\d]*([\d.]+\s*%)'),
            "Uniformity": re.compile(r'Match Uniformity[^\d]*([\d.]+\s*%?(?:\s*covered)?)')
        }

    def setup_ui(self):
        # Apply deep space / research aesthetic theme with new fonts
        self.setStyleSheet("""
            QWidget {
                background-color: #0b0f19;
                color: #e0e0e0;
                font-family: 'Roboto', 'Inter', 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #1a2333;
                border: 1px solid #2a3b5c;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                color: #66fcf1;
            }
            QPushButton:hover {
                background-color: #24344d;
                border: 1px solid #45a29e;
            }
            QPushButton:disabled {
                color: #4b5563;
                background-color: #111827;
                border: 1px solid #1f2937;
            }
            QListWidget {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #1d4ed8;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #c9d1d9;
            }
            QLabel {
                background-color: transparent;
            }
            QFrame#MetricsPanel {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 6px;
            }
            QLabel#ImageLabel {
                border: 1px solid #1f2937;
                background-color: #111827;
            }
            QLabel#MetricTitle {
                font-size: 14px;
                font-weight: bold;
                color: #45a29e;
                border-bottom: 1px solid #1f2937;
                padding-bottom: 5px;
            }
            QLabel#MetricName {
                color: #9ca3af;
                font-size: 12px;
            }
            QLabel#MetricValue {
                color: #e5e7eb;
                font-size: 12px;
                font-weight: bold;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Top Splitter (Left: Folder/List, Right: Image)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Top Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_select_folder = QPushButton("Select Folder")
        self.btn_select_folder.clicked.connect(self.select_folder)
        
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(120, 120))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setWordWrap(True)
        
        left_layout.addWidget(self.btn_select_folder)
        left_layout.addWidget(self.list_widget)
        
        # --- Top Right Panel ---
        self.lbl_image = QLabel("Output Result\n\n(Select an image and run search)")
        self.lbl_image.setObjectName("ImageLabel")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(500, 400)
        self.lbl_image.setStyleSheet("font-size: 18px; color: #4b5563; font-weight: bold;")
        
        top_splitter.addWidget(left_panel)
        top_splitter.addWidget(self.lbl_image)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)
        
        # --- Bottom Panel ---
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QFont("Consolas", 10))
        self.txt_output.setPlaceholderText("Terminal output will appear here...")
        
        # Right side of bottom panel (Button + Metrics)
        bottom_right_panel = QWidget()
        bottom_right_layout = QVBoxLayout(bottom_right_panel)
        bottom_right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_register = QPushButton("Register Search")
        self.btn_register.setMinimumHeight(50)
        self.btn_register.setStyleSheet("font-size: 16px;")
        self.btn_register.clicked.connect(self.run_registration)
        
        # Metrics Panel
        self.metrics_frame = QFrame()
        self.metrics_frame.setObjectName("MetricsPanel")
        metrics_layout = QVBoxLayout(self.metrics_frame)
        
        title_lbl = QLabel("Evaluation Metrics")
        title_lbl.setObjectName("MetricTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(title_lbl)
        
        grid = QGridLayout()
        grid.setSpacing(10)
        self.metrics_labels = {}
        
        metric_names = [
            ("RMSE", "RMSE (Accuracy)"),
            ("PSNR", "PSNR (Radiometric)"),
            ("SSIM", "SSIM (Structural)"),
            ("Inlier Ratio", "Inlier Ratio"),
            ("Uniformity", "Match Uniformity")
        ]
        
        for i, (key, display_name) in enumerate(metric_names):
            name_lbl = QLabel(display_name)
            name_lbl.setObjectName("MetricName")
            
            val_lbl = QLabel("-")
            val_lbl.setObjectName("MetricValue")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            grid.addWidget(name_lbl, i, 0)
            grid.addWidget(val_lbl, i, 1)
            self.metrics_labels[key] = val_lbl
            
        metrics_layout.addLayout(grid)
        metrics_layout.addStretch()
        
        bottom_right_layout.addWidget(self.btn_register)
        bottom_right_layout.addWidget(self.metrics_frame)
        
        bottom_layout.addWidget(self.txt_output, stretch=4)
        bottom_layout.addWidget(bottom_right_panel, stretch=1)
        
        # Add to main layout
        main_layout.addWidget(top_splitter, stretch=2)
        main_layout.addWidget(bottom_panel, stretch=1)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            self.list_widget.clear()
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    filepath = os.path.join(folder, filename)
                    icon = QIcon(filepath)
                    item = QListWidgetItem(icon, filename)
                    item.setData(Qt.ItemDataRole.UserRole, filepath)
                    self.list_widget.addItem(item)
                    
    def reset_metrics(self):
        for val_lbl in self.metrics_labels.values():
            val_lbl.setText("-")

    def run_registration(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            self.txt_output.insertHtml("<span style='color: #ff5555;'>Error: Please select an image from the list first.</span><br>")
            return
            
        query_img_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        isro_img = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/data/calibrated/20260102/ch2_ohr_ncp_20260102T1819015920_d_img_d18.img"
        isro_xml = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/data/calibrated/20260102/ch2_ohr_ncp_20260102T1819015920_d_img_d18.xml"
        index_h5 = "/home/friday/hhh/reference_features.h5"
        output_png = "/home/friday/helios1/outputs/ui_result.png"
        
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        
        self.btn_register.setEnabled(False)
        self.txt_output.clear()
        self.reset_metrics()
        self.stdout_buffer = ""
        self.txt_output.insertHtml(f"<span style='color: #8be9fd;'>Starting registration for: {os.path.basename(query_img_path)}...</span><br>")
        
        self.lbl_image.setText("Running registration...\nPlease wait.")
        self.lbl_image.setPixmap(QPixmap()) 
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.process = QProcess(self)
        self.process.setWorkingDirectory(project_root)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        
        venv_python = os.path.join(project_root, ".venv", "bin", "python")
        python_exec = venv_python if os.path.exists(venv_python) else "python"
        
        env = self.process.processEnvironment()
        env.insert("FORCE_COLOR", "1")
        self.process.setProcessEnvironment(env)
        
        args = [
            "main.py", "register",
            "--query-img", query_img_path,
            "--isro-img", isro_img,
            "--isro-xml", isro_xml,
            "--index-h5", index_h5,
            "--output", output_png
        ]
        
        self.process.start(python_exec, args)

    def extract_metrics(self, raw_text):
        # Strip ANSI codes for regex matching
        clean_text = re.sub(r'\x1b\[[\d;]*m', '', raw_text)
        
        for key, pattern in self.metric_patterns.items():
            match = pattern.search(clean_text)
            if match:
                self.metrics_labels[key].setText(match.group(1).strip())

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        
        # Accumulate buffer for metric extraction in case a line is split
        self.stdout_buffer += data
        self.extract_metrics(self.stdout_buffer)
        
        html_data = ansi_to_html(data)
        self.txt_output.insertHtml(html_data)
        scrollbar = self.txt_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        html_data = ansi_to_html(data)
        if not html_data.startswith("<span"):
            html_data = f"<span style='color: #ff5555;'>{html_data}</span>"
        self.txt_output.insertHtml(html_data)
        scrollbar = self.txt_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_finished(self):
        self.btn_register.setEnabled(True)
        self.txt_output.insertHtml("<br><br><span style='color: #50fa7b;'>Process finished.</span>")
        
        output_png = "/home/friday/helios1/outputs/ui_result.png"
        if os.path.exists(output_png):
            pixmap = QPixmap(output_png)
            scaled_pixmap = pixmap.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_image.setPixmap(scaled_pixmap)
            self.lbl_image.setText("")
        else:
            self.lbl_image.setText("Result image not found.\nCheck CLI output for errors.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
