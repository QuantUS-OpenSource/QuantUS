"""
Config Preview Widget

This widget displays the loaded configuration parameters and allows users
to review and confirm them before proceeding with the analysis.
"""

import os
import pickle
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QMessageBox, QFileDialog

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.config_loading.ui.config_preview_ui import Ui_configPreview
from engines.qus.quantus.data_objs import RfAnalysisConfig, UltrasoundRfImage, BmodeSeg


class ConfigPreviewWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming analysis configuration.
    
    Displays all configuration parameters in a readable format
    and allows users to confirm or go back to modify them.
    """
    
    # ============================================================================
    # SIGNALS
    # ============================================================================
    
    config_confirmed = pyqtSignal(RfAnalysisConfig)  # config_data
    back_requested = pyqtSignal()
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    def __init__(self, image_data: UltrasoundRfImage = None, seg_data: BmodeSeg = None, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_configPreview()
        self._config_data: Optional[RfAnalysisConfig] = None
        self._image_data = image_data
        self._seg_data = seg_data
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout
        self.setLayout(self._ui.full_screen_layout)

        # Organize menu objects
        self._save_config_menu_objects = [
            'dest_folder_label', 'save_folder_input',
            'choose_save_folder_button', 'clear_save_folder_button',
            'config_name_label', 'save_name_input',
            'save_config_button', 'back_from_save_button'
        ]
        self._hide_save_menu()
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.config_confirm_layout, 10)
        
        # Update labels to reflect inputted image and phantom
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._ui.confirm_config_button.clicked.connect(self._on_confirm_clicked)
        self._ui.save_option_button.clicked.connect(self._on_save_menu_clicked)
        self._ui.back_from_save_button.clicked.connect(self._on_back_from_save)
        self._ui.save_config_button.clicked.connect(self._on_save_config)
        self._ui.choose_save_folder_button.clicked.connect(self._select_dest_folder)
        self._ui.clear_save_folder_button.clicked.connect(self._ui.save_folder_input.clear)

    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    def set_config_data(self, config_data: RfAnalysisConfig) -> None:
        """
        Set the configuration data to display.
        
        Args:
            config_data: Analysis configuration data
        """
        self._config_data = config_data
        self._update_display()

    def show_error(self, error_message: str) -> None:
        """
        Show error message.
        
        Args:
            error_message: Error message to display
        """
        QMessageBox.critical(self, "Error", error_message)
    
    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    def _hide_save_menu(self) -> None:
        """Hide the save configuration menu."""
        for obj_name in self._save_config_menu_objects:
            obj = getattr(self._ui, obj_name, None)
            if obj:
                obj.hide()
            else:
                print(f"Warning: Could not find UI object '{obj_name}' to hide it.")
    
    def _show_save_menu(self) -> None:
        """Show the save configuration menu."""
        for obj_name in self._save_config_menu_objects:
            obj = getattr(self._ui, obj_name, None)
            if obj:
                obj.show()
            else:
                print(f"Warning: Could not find UI object '{obj_name}' to show it.")

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()

    def _on_save_menu_clicked(self) -> None:
        """Handle save menu button click."""
        self._show_save_menu()
        self._ui.save_option_button.hide()

    def _on_back_from_save(self) -> None:
        """Handle back button click from save menu."""
        self._hide_save_menu()
        self._ui.save_option_button.show()

    def _on_save_config(self) -> None:
        """Handle save configuration button click."""
        save_path = Path(self._ui.save_folder_input.text())
        if not save_path.exists():
            self.show_error("Error: Save folder does not exist.")
            return
        save_pathname = self._ui.save_name_input.text()
        if not save_pathname:
            self.show_error("Error: Please provide a name for the configuration file.")
            return
        
        full_save_path = save_path / f"{save_pathname}.pkl"
        full_save_path = full_save_path.with_suffix('.pkl')

        try:
            with open(full_save_path, 'wb') as f:
                pickle.dump({
                    "Config": self._config_data,
                    "Image Name": self._image_data.scan_name,
                    "Phantom Name": self._image_data.phantom_name,
                }, f)
            QMessageBox.information(self, "Success", f"Configuration saved to {full_save_path}")
            self._hide_save_menu()
        except Exception as e:
            self.show_error(f"Error saving configuration: {e}")

    def _on_confirm_clicked(self) -> None:
        """Handle confirm button click."""
        if self._config_data:
            self.config_confirmed.emit(self._config_data)
        else:
            print("Config Preview Error: No configuration data to confirm")
            
    def _update_display(self) -> None:
        """Update the display with current configuration data."""
        if not self._config_data:
            return
            
        # Update frequency parameters
        self._ui.transducerFreqValue.setText(str(self._config_data.transducer_freq_band))
        self._ui.analysisFreqValue.setText(str(self._config_data.analysis_freq_band))
        self._ui.centerFreqValue.setText(str(self._config_data.center_frequency))
        self._ui.samplingFreqValue.setText(str(self._config_data.sampling_frequency))
        self._ui.axWinSizeValue.setText(str(self._config_data.ax_win_size))
        self._ui.latWinSizeValue.setText(str(self._config_data.lat_win_size))
        self._ui.windowThreshValue.setText(str(self._config_data.window_thresh*100))
        self._ui.axialOverlapValue.setText(str(self._config_data.axial_overlap*100))
        self._ui.lateralOverlapValue.setText(str(self._config_data.lateral_overlap*100))
            
        # Update 3D parameters
        if self._image_data.spatial_dims == 3:
            self._ui.corWinSizeValue.setText(str(self._config_data.cor_win_size))
            self._ui.coronalOverlapValue.setText(str(self._config_data.coronal_overlap*100))
        else:
            self._ui.group3d.hide()

    def _select_dest_folder(self) -> None:
        """
        Helper method for folder selection dialogs.

        Args:
            path_input: QLineEdit widget to update with selected folder path
        """
        # Check if folder path is manually typed and exists
        if os.path.isdir(self._ui.save_folder_input.text()):
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder"
        )

        if folder:
            self._ui.save_folder_input.setText(folder)