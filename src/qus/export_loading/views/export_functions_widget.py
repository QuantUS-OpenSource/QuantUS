"""
Export Functions Widget for QuantUS GUI

This widget allows users to select export functions with recommendations
based on the visualization results.
"""

from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QListWidget, QListWidgetItem, QGroupBox, QAbstractItemView)
from PyQt6.QtCore import pyqtSignal, Qt


class ExportFunctionsWidget(QWidget):
    """Widget for selecting export functions."""
    
    # Signals
    export_functions_selected = pyqtSignal(list)  # Emitted when export functions are selected
    
    def __init__(self, controller, parent=None):
        """Initialize the export functions widget.
        
        Args:
            controller: The export loading controller.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.controller = controller
        
        # UI elements
        self._function_list = None
        self._recommended_label = None
        self._available_functions = {}
        self._recommended_functions = []
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Export Functions Group Box
        functions_group = QGroupBox("Export Functions")
        functions_group.setStyleSheet("""
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
        
        functions_layout = QVBoxLayout()
        functions_group.setLayout(functions_layout)
        
        # Recommended functions section
        self._recommended_label = QLabel("Automatically selected export functions:")
        self._recommended_label.setWordWrap(True)
        self._recommended_label.setStyleSheet("color: green; font-weight: bold;")
        functions_layout.addWidget(self._recommended_label)
        
        # Function selection list
        function_layout = QHBoxLayout()
        function_label = QLabel("Available Functions:")
        function_label.setStyleSheet("color: rgb(255, 255, 255); font-size: 14px;")
        
        self._function_list = QListWidget()
        self._function_list.setStyleSheet("""
            QListWidget {
                color: rgb(128, 128, 128);
                background-color: rgb(40, 40, 40);
                border: 1px solid rgb(80, 80, 80);
                border-radius: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid rgb(60, 60, 60);
            }
            QListWidget::item:selected {
                background-color: rgb(60, 60, 60);
            }
            QListWidget::item:hover {
                background-color: rgb(50, 50, 50);
            }
        """)
        self._function_list.setEnabled(False)  # Disable the list since functions are auto-selected
        
        function_layout.addWidget(function_label)
        function_layout.addWidget(self._function_list)
        
        functions_layout.addLayout(function_layout)
        
        # Instructions
        instructions_label = QLabel("Export functions are automatically selected: descr_vals for all analyses, hscan_stats for hscan analysis.")
        instructions_label.setWordWrap(True)
        instructions_label.setStyleSheet("color: green; font-style: italic; font-weight: bold;")
        functions_layout.addWidget(instructions_label)
        
        main_layout.addWidget(functions_group)
    
    def _connect_signals(self):
        """Connect signals."""
        if self._function_list:
            self._function_list.itemSelectionChanged.connect(self._on_function_selection_changed)
    
    def _on_function_selection_changed(self):
        """Handle function selection change."""
        selected_functions = []
        for i in range(self._function_list.count()):
            item = self._function_list.item(i)
            if item.isSelected():
                selected_functions.append(item.text())
        
        self.export_functions_selected.emit(selected_functions)
    
    def update_available_functions(self, functions: Dict[str, Any]):
        """Update the available export functions.
        
        Args:
            functions: Dictionary of available export functions.
        """
        self._available_functions = functions
        
        if self._function_list:
            self._function_list.clear()
            
            for func_name in sorted(functions.keys()):
                item = QListWidgetItem(func_name)
                self._function_list.addItem(item)
    
    def set_recommended_functions(self, recommended_functions: List[str]):
        """Set the automatically selected export functions.
        
        Args:
            recommended_functions: List of automatically selected function names.
        """
        self._recommended_functions = recommended_functions
        
        if self._recommended_label:
            if recommended_functions:
                recommended_text = f"Automatically selected: {', '.join(recommended_functions)}"
                self._recommended_label.setText(recommended_text)
                self._recommended_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self._recommended_label.setText("No functions automatically selected.")
                self._recommended_label.setStyleSheet("color: orange; font-weight: bold;")
    
    def set_selected_functions(self, selected_functions: List[str]):
        """Set the selected export functions.
        
        Args:
            selected_functions: List of selected function names.
        """
        if self._function_list:
            # Clear current selection
            self._function_list.clearSelection()
            
            # Select the specified functions
            for i in range(self._function_list.count()):
                item = self._function_list.item(i)
                if item.text() in selected_functions:
                    item.setSelected(True)
