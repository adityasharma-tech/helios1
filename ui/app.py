import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QLabel, 
                             QTextEdit, QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QPixmap, QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helios Image Registration Pipeline")
        self.resize(1000, 700)
        self.setup_ui()
        self.process = None

    def setup_ui(self):
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #2c2c2c;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
                border: 1px solid #6a6a6a;
            }
            QListWidget, QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 4px;
            }
            QLabel {
                border: 1px solid #333333;
                background-color: #1a1a1a;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top Splitter (Left: Folder/List, Right: Image)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Top Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_select_folder = QPushButton("Select Folder")
        self.btn_select_folder.clicked.connect(self.select_folder)
        
        self.list_widget = QListWidget()
        
        left_layout.addWidget(self.btn_select_folder)
        left_layout.addWidget(self.list_widget)
        
        # --- Top Right Panel ---
        self.lbl_image = QLabel("Output Result\n\n(Select an image and run search)")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(400, 300)
        self.lbl_image.setStyleSheet("font-size: 18px; color: #888888;")
        
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
        self.txt_output.setPlaceholderText("CLI Output...")
        
        self.btn_register = QPushButton("Register Search")
        self.btn_register.setMinimumHeight(40)
        self.btn_register.clicked.connect(self.run_registration)
        
        bottom_layout.addWidget(self.txt_output, stretch=4)
        bottom_layout.addWidget(self.btn_register, stretch=1)
        
        # Add to main layout
        main_layout.addWidget(top_splitter, stretch=2)
        main_layout.addWidget(bottom_panel, stretch=1)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            self.list_widget.clear()
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.list_widget.addItem(os.path.join(folder, filename))

    def run_registration(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            self.txt_output.append("Error: Please select an image from the list first.")
            return
            
        query_img_path = selected_items[0].text()
        
        # Hardcoded paths based on previous CLI example
        isro_img = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/data/calibrated/20260102/ch2_ohr_ncp_20260102T1819015920_d_img_d18.img"
        isro_xml = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/data/calibrated/20260102/ch2_ohr_ncp_20260102T1819015920_d_img_d18.xml"
        index_h5 = "/home/friday/hhh/reference_features.h5"
        output_png = "/home/friday/helios1/outputs/ui_result.png"
        
        # Ensure outputs directory exists
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        
        self.btn_register.setEnabled(False)
        self.txt_output.clear()
        self.txt_output.append(f"Starting registration for: {os.path.basename(query_img_path)}...\n")
        
        self.lbl_image.setText("Running registration...\nPlease wait.")
        self.lbl_image.setPixmap(QPixmap()) # clear image
        
        # Change directory to the main project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.process = QProcess(self)
        self.process.setWorkingDirectory(project_root)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        
        # Use python from the virtualenv if it exists
        venv_python = os.path.join(project_root, ".venv", "bin", "python")
        python_exec = venv_python if os.path.exists(venv_python) else "python"
        
        args = [
            "main.py", "register",
            "--query-img", query_img_path,
            "--isro-img", isro_img,
            "--isro-xml", isro_xml,
            "--index-h5", index_h5,
            "--output", output_png
        ]
        
        self.process.start(python_exec, args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8')
        self.txt_output.append(data.strip())
        scrollbar = self.txt_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8')
        # Use raw string to avoid rich formatting escaping issues, but display in UI
        self.txt_output.append(data.strip())
        scrollbar = self.txt_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_finished(self):
        self.btn_register.setEnabled(True)
        self.txt_output.append("\nProcess finished.")
        
        output_png = "/home/friday/helios1/outputs/ui_result.png"
        if os.path.exists(output_png):
            pixmap = QPixmap(output_png)
            # Scale pixmap to fit label while keeping aspect ratio
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
