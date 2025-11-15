"""
Segmentation File Selection Widget for Segmentation Loading
"""

import os
from typing import Optional
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.seg_file_selection_ui import Ui_constructRoi
from engines.qus.quantus.data_objs import UltrasoundRfImage


class SegFileSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting segmentation files.
    
    This is the second step in the segmentation loading process where users
    select the actual segmentation files to load.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    file_selected = pyqtSignal(dict)  # {'seg_path': str, 'seg_type': str}
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, seg_type_name: str, image_data: UltrasoundRfImage, file_extensions: list = None, loading_options: list = None, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._seg_type_name = seg_type_name
        self._image_data = image_data
        self._file_extensions = file_extensions or []
        self._loading_options = loading_options or []
        self._loading_widgets: list = []

        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for file selection only - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors for file selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.seg_loading_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        
        # Store widgets that should be disabled during loading
        self._loading_widgets = [
            self._ui.choose_seg_path_button,
            self._ui.clear_seg_path_button,
            self._ui.accept_seg_path_button,
            self._ui.back_button
        ]

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)
        
        # Update labels to reflect selected segmentation type
        self._ui.select_seg_label.setText(f"Select {self._seg_type_name} Segmentation")
        self._ui.seg_path_label.setText(f"Input Path to Segmentation file\n({', '.join(self._file_extensions)})")

        if len(self._loading_options):
            self._show_loading_options()
        else:
            self._hide_loading_options()
        self.hide_loading()
        self.clear_error()

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.choose_seg_path_button.clicked.connect(self._on_choose_seg_path)
        self._ui.clear_seg_path_button.clicked.connect(self._ui.seg_path_input.clear)
        self._ui.accept_seg_path_button.clicked.connect(self._on_load_segmentation)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def update_display(self, data) -> None:
        """Update the view with new data."""
        # This widget doesn't need to update with external data
        pass
        
    def get_seg_path(self) -> str:
        """Get the selected segmentation file path."""
        return self._ui.seg_path_input.text().strip()
        
    def show_loading(self) -> None:
        """Show loading state in the UI."""
        super().show_loading()
        # Disable widgets during loading
        for widget in self._loading_widgets:
            if hasattr(widget, 'setEnabled'):
                widget.setEnabled(False)

        # Show loading message after a small delay
        self._loading_timer = getattr(self, '_loading_timer', None)
        if self._loading_timer:
            self._loading_timer.stop()
        
        self._loading_timer = QTimer()
        self._loading_timer.singleShot(200, self._show_loading_message)  # 200ms delay

    def hide_loading(self) -> None:
        """Hide loading state in the UI."""
        super().hide_loading()

        # Cancel loading timer if it exists
        loading_timer = getattr(self, '_loading_timer', None)
        if loading_timer:
            loading_timer.stop()

        # Re-enable widgets after loading
        for widget in self._loading_widgets:
            if hasattr(widget, 'setEnabled'):
                widget.setEnabled(True)

        self._ui.loading_screen_label.hide()

    def _show_loading_message(self) -> None:
        """Show the loading message after delay."""
        # Only show if we're still in loading state
        if self._is_loading:
            self._ui.loading_screen_label.show()
        
    def show_error(self, error_message: str) -> None:
        """
        Display error message to user.
        
        Args:
            error_message: Error message to display
        """
        self._ui.select_seg_error_msg.setText(error_message)
        self._ui.select_seg_error_msg.show()
        
    def clear_error(self) -> None:
        """Clear any displayed error message."""
        self._ui.select_seg_error_msg.clear()
        self._ui.select_seg_error_msg.hide()
            
    def _on_choose_seg_path(self) -> None:
        """Handle segmentation file selection."""
        self._select_file_helper(self._ui.seg_path_input, self._file_extensions)
        
    def _on_load_segmentation(self) -> None:
        """Handle segmentation loading request."""
        seg_path = self.get_seg_path()
        
        if not seg_path:
            self.show_error("Please select a segmentation file")
            return
            
        if not os.path.exists(seg_path):
            self.show_error(f"Segmentation file does not exist: {os.path.basename(seg_path)}")
            return
            
        if self._file_extensions and not seg_path.endswith(tuple(self._file_extensions)):
            self.show_error(f"Segmentation file must have one of the following extensions: {', '.join(self._file_extensions)}")
            return
            
        self.clear_error()
        
        self.file_selected.emit({
            'seg_path': seg_path,
            'seg_type': self._seg_type_name
        })
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.clear_error()
        self.back_requested.emit()
        
    def _select_file_helper(self, path_input, file_exts: list) -> None:
        """
        Helper method for file selection dialogs.
        
        Args:
            path_input: QLineEdit widget to update with selected path
            file_exts: List of file extensions for filtering
        """
        # Check if file path is manually typed and exists
        if os.path.exists(path_input.text()):
            return
            
        # Create filter string
        if file_exts:
            filter_str = " ".join([f"*{ext}" for ext in file_exts])
        else:
            filter_str = "All Files (*)"
            
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Segmentation File", 
            filter=filter_str
        )
        
        if file_name:
            path_input.setText(file_name)

    def _show_loading_options(self) -> None:
        """
        Show additional loading options if available.
        """
        if self._loading_options:
            self._ui.loading_options_table.clear()
            self._ui.loading_options_table.setRowCount(len(self._loading_options))
            self._ui.loading_options_table.setColumnCount(2)
            self._ui.loading_options_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            self._ui.loading_options_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self._ui.loading_options_table.setColumnWidth(0, self._ui.loading_options_table.viewport().width() * 3)
            self._ui.loading_options_table.setHorizontalHeaderLabels(["Option", "Value"])
            for row_ix, option in enumerate(self._loading_options):
                item = QTableWidgetItem(option.capitalize())
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsSelectable)
                self._ui.loading_options_table.setItem(row_ix, 0, item)
                self._ui.loading_options_table.setItem(row_ix, 1, QTableWidgetItem(""))

    def _hide_loading_options(self) -> None:
        """
        Hide the loading options table.
        """
        self._ui.loading_options_table.clear()
        self._ui.loading_options_table.setRowCount(0)
        self._ui.loading_options_table.setColumnCount(0)
        self._ui.loading_options_table.setHorizontalHeaderLabels([])
        self._ui.loading_options_table.hide()
        self._ui.loading_options_label.hide()
