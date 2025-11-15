"""
Config File Selection Widget for Config Loading

This widget allows users to select configuration files to load.
It provides file browsing functionality and handles file selection.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox, QTableWidgetItem, \
    QHeaderView

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.config_loading.ui.config_file_selection_ui import Ui_configFileSelection
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


class ConfigFileSelectionWidget(QWidget, BaseViewMixin):
    """
    Widget for selecting configuration files to load.
    
    This is the file selection step in the config loading process where users
    choose a configuration file to load. Designed to be used within the main 
    application widget stack.
    """
    
    # Signals for communicating with controller
    file_selected = pyqtSignal(dict)  # file_data
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, config_type_name: str, image_data: UltrasoundRfImage, seg_data: BmodeSeg, file_extensions: list = None,
                  loading_options: list = None, default_option_vals: list = None, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        assert len(loading_options or []) == len(default_option_vals or []), \
            "Loading options and default values must be of the same length."

        self._ui = Ui_configFileSelection()
        self._config_type_name = config_type_name
        self._image_data = image_data
        self._seg_data = seg_data
        self._file_extensions: List[str] = file_extensions or []
        self._loading_options: List[str] = loading_options or []
        self._default_option_vals: List[str] = default_option_vals or []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for file selection only
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors for file selection
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.config_loading_layout, 10)

        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        # Update labels to reflect selected config type
        self._ui.file_selection_label.setText(f"Select {self._config_type_name} Configuration")
        self._ui.config_path_label.setText(f"Input path to configuration file\n({', '.join(self._file_extensions)})")

        if len(self._loading_options):
            self._show_loading_options()
        else:
            self._hide_loading_options()

    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.choose_config_path_button.clicked.connect(self._on_choose_config_path)
        self._ui.clear_config_path_button.clicked.connect(self._ui.config_path_input.clear)
        self._ui.accept_config_path_button.clicked.connect(self._on_config_accepted)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def show_error(self, error_message: str) -> None:
        """
        Show error message.
        
        Args:
            error_message: Error message to display
        """
        QMessageBox.critical(self, "Error", error_message)
        
    # ============================================================================
    # EVENT HANDLERS
    # ============================================================================

    def _on_config_accepted(self) -> None:
        """Handle file selection."""

        # Collect config_loader_kwargs from loading options table
        config_loader_kwargs = {}
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

                    config_loader_kwargs[option] = value

        selected_file_path = self._ui.config_path_input.text().strip()
        if selected_file_path:
            # Create file data dictionary
            file_data = {
                'file_path': selected_file_path,
                'config_loader_kwargs': config_loader_kwargs
            }

            if not os.path.exists(selected_file_path):
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"The selected file does not exist:\n{selected_file_path}"
                )
                return

            self.file_selected.emit(file_data)
        else:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select a configuration file first."
            )
            
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    def _on_choose_config_path(self) -> None:
        """Handle configuration file selection."""
        self._select_file_helper(self._ui.config_path_input, self._file_extensions)

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
            "Open Configuration File", 
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
                self._ui.loading_options_table.setItem(row_ix, 1, QTableWidgetItem(self._default_option_vals[row_ix]))

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
