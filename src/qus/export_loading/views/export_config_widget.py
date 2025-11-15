"""
Export Configuration Widget for QuantUS GUI

This widget allows users to configure export settings including export type and path.
"""

from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QLineEdit, QPushButton, QGroupBox, QFormLayout)
from PyQt6.QtCore import pyqtSignal, Qt


class ExportConfigWidget(QWidget):
    """Widget for configuring export settings."""
    
    # Signals
    export_path_changed = pyqtSignal(str)  # Emitted when export path changes
    
    def __init__(self, controller, parent=None):
        """Initialize the export configuration widget.
        
        Args:
            controller: The export loading controller.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.controller = controller
        
        # UI elements
        self._path_edit = None
        self._browse_button = None
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Export Configuration Group Box
        config_group = QGroupBox("Export Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                color: rgb(255, 255, 255);
                font-size: 16px;
                font-weight: bold;
                border: 2px solid rgb(99, 0, 174);
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        config_layout = QFormLayout()
        config_group.setLayout(config_layout)
        
        # CSV Export Type Note
        csv_note = QLabel("Export Type: CSV (always selected)")
        csv_note.setStyleSheet("color: rgb(128, 128, 128); font-size: 14px; font-style: italic;")
        config_layout.addRow("", csv_note)
        
        # Export Path
        path_label = QLabel("Export Path:")
        path_label.setStyleSheet("color: rgb(255, 255, 255); font-size: 14px;")
        
        path_layout = QHBoxLayout()
        
        self._path_edit = QLineEdit()
        self._path_edit.setStyleSheet("""
            QLineEdit {
                color: rgb(255, 255, 255);
                background-color: rgb(60, 60, 60);
                border: 1px solid rgb(99, 0, 174);
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        
        self._browse_button = QPushButton("Browse")
        self._browse_button.setStyleSheet("""
            QPushButton {
                color: white;
                font-size: 14px;
                background: rgb(90, 37, 255);
                border-radius: 10px;
                padding: 5px 15px;
            }
        """)
        
        path_layout.addWidget(self._path_edit)
        path_layout.addWidget(self._browse_button)
        
        config_layout.addRow(path_label, path_layout)
        
        main_layout.addWidget(config_group)
    
    def _connect_signals(self):
        """Connect signals."""
        if self._path_edit:
            self._path_edit.textChanged.connect(self._on_path_changed)
        
        if self._browse_button:
            self._browse_button.clicked.connect(self._on_browse_clicked)
    
    def _on_path_changed(self, path: str):
        """Handle export path change."""
        self.export_path_changed.emit(path)
    
    def _on_browse_clicked(self):
        """Handle browse button click."""
        from PyQt6.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Select export file", 
            "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if path:
            self._path_edit.setText(path)
            self.export_path_changed.emit(path)
    
    def set_export_path(self, path: str):
        """Set the export path.
        
        Args:
            path: The export path to set.
        """
        if self._path_edit:
            self._path_edit.setText(path)
