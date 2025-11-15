"""
File Selection Widget for Image Loading
"""

import os
from typing import Optional
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


from src.qus.mvc.base_view import BaseViewMixin
from src.qus.image_loading.ui.file_selection_ui import Ui_selectImage


class FileSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting image and phantom files.
    
    This is the second step in the image loading process where users
    select the actual files to load after choosing the scan type.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    files_selected = pyqtSignal(dict)  # {'image_path': str, 'phantom_path': str, 'scan_loader_kwargs': dict}
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, scan_type_name: str, file_extensions: list = None, loading_options: list = None, default_vals: list = None, 
                 parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_selectImage()
        self._scan_type_name = scan_type_name
        self._file_extensions = file_extensions or []
        self._loading_options = loading_options or []
        self._default_vals = default_vals or []
        self._loading_widgets: list = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for file selection only
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors for file selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.img_selection_layout, 6)
        
        # Store widgets that should be disabled during loading
        self._loading_widgets = [
            self._ui.choose_image_path_button,
            self._ui.choose_phantom_path_button,
            self._ui.clear_image_path_button,
            self._ui.clear_phantom_path_button,
            self._ui.generate_image_button,
            self._ui.back_button
        ]
        
        # Update labels to reflect selected scan type
        self._ui.select_data_label.setText(f"Select {self._scan_type_name} Files")
        self._ui.image_path_label.setText(f"Input Path to Image file\n ({', '.join(self._file_extensions)})")
        self._ui.phantom_path_label.setText(f"Input Path to Phantom file\n ({', '.join(self._file_extensions)})")

        if len(self._loading_options):
            self._show_loading_options()
        else:
            self._hide_loading_options()

        self.hide_loading()
        self.clear_error()
        
    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.choose_image_path_button.clicked.connect(self._on_choose_image_path)
        self._ui.choose_phantom_path_button.clicked.connect(self._on_choose_phantom_path)
        self._ui.clear_image_path_button.clicked.connect(self._ui.image_path_input.clear)
        self._ui.clear_phantom_path_button.clicked.connect(self._ui.phantom_path_input.clear)
        self._ui.generate_image_button.clicked.connect(self._on_generate_image)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
    
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
                self._ui.loading_options_table.setItem(row_ix, 1, QTableWidgetItem(self._default_vals[row_ix]))

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
                
    def _show_loading_message(self) -> None:
        """Show the loading message after delay."""
        # Only show if we're still in loading state
        if self._is_loading:
            self._ui.loading_screen_label.show()

    def _on_choose_image_path(self) -> None:
        """Handle image file selection."""
        self._select_file_helper(self._ui.image_path_input, self._file_extensions)
        
    def _on_choose_phantom_path(self) -> None:
        """Handle phantom file selection."""
        self._select_file_helper(self._ui.phantom_path_input, self._file_extensions)
        
    def _on_generate_image(self) -> None:
        """Handle image generation request."""
        image_path = self._ui.image_path_input.text().strip()
        phantom_path = self._ui.phantom_path_input.text().strip()

        if not image_path:
            self.show_error("Please select an image file")
            return
        if not phantom_path:
            self.show_error("Please select a phantom file")
            return
        if not os.path.exists(image_path):
            self.show_error(f"Image file does not exist: {os.path.basename(image_path)}")
            return
        if not os.path.exists(phantom_path):
            self.show_error(f"Phantom file does not exist: {os.path.basename(phantom_path)}")
            return
        if not image_path.endswith(tuple(self._file_extensions)):
            self.show_error(f"Image file must have one of the following extensions: {', '.join(self._file_extensions)}")
            return
        if not phantom_path.endswith(tuple(self._file_extensions)):
            self.show_error(f"Phantom file must have one of the following extensions: {', '.join(self._file_extensions)}")
            return
            
        self.clear_error()
        
        # Collect scan_loader_kwargs from loading options table
        scan_loader_kwargs = {}
        if self._loading_options:
            table = self._ui.loading_options_table
            for option, row in zip(self._loading_options, range(table.rowCount())):
                value_item = table.item(row, 1)
                if value_item is not None and (inputted_text := value_item.text().strip()) != "":
                    inputted_text = inputted_text if inputted_text != "false" else "False"
                    inputted_text = inputted_text if inputted_text != "true" else "True"
                    try:
                        value = eval(inputted_text)
                    except NameError:
                        value = inputted_text
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating loading option '{option}': {e}")
                    
                    scan_loader_kwargs[option] = value
        
        self.files_selected.emit({
            'image_path': image_path,
            'phantom_path': phantom_path,
            'scan_loader_kwargs': scan_loader_kwargs
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
            "Open File", 
            filter=filter_str
        )
        
        if file_name:
            path_input.setText(file_name)

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
        
    def show_error(self, error_message: str) -> None:
        """
        Display error message to user.
        
        Args:
            error_message: Error message to display
        """
        self._ui.select_image_error_msg.setText(error_message)
        self._ui.select_image_error_msg.show()
        
    def clear_error(self) -> None:
        """Clear any displayed error message."""
        self._ui.select_image_error_msg.clear()
        self._ui.select_image_error_msg.hide()
