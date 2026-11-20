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
            
    # ------------------------------------------------------------------ window sizes in both units
    def _roi_line_spacing_mm(self) -> Optional[float]:
        """Mean distance (mm) between neighbouring A-lines inside the ROI: the probe pitch for a linear
        array; for a curvilinear array the spacing grows with depth, so this is its mean over the ROI.
        None if it cannot be computed (no 2D mask / positions)."""
        if hasattr(self, "_line_spacing_cache"):
            return self._line_spacing_cache
        spacing = None
        try:
            import numpy as np
            from engines.qus.quantus.seg_loading.correlation_length import positions_from_image
            rf = self._image_data.rf_data
            frame_shape = rf.shape[1:] if rf.ndim == 3 else rf.shape
            mask = np.asarray(getattr(self._seg_data, "seg_mask", None), dtype=bool)
            if mask.ndim == 2 and mask.shape == tuple(frame_shape):
                ax, lat = positions_from_image(self._image_data, *frame_shape)
                d = np.hypot(np.diff(lat, axis=1), np.diff(ax, axis=1))
                inside = mask[:, 1:] & mask[:, :-1]
                if inside.any():
                    spacing = float(np.mean(d[inside]))
        except Exception:
            spacing = None
        self._line_spacing_cache = spacing
        return spacing

    def _axial_size_text(self, cfg) -> str:
        """'2.2 mm (15.7 wavelengths)': mm and wavelengths (c / centre frequency), as the analysis."""
        size, units = cfg.ax_win_size, getattr(cfg, "ax_win_size_units", "mm")
        try:
            wavelength_mm = float(cfg.speed_of_sound) / float(cfg.center_frequency) * 1000
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            return _with_unit(_num(size), _unit_label(units))
        if units == "number of wavelength":
            return f"{_num(size * wavelength_mm)} mm ({_num(size)} wavelengths)"
        return f"{_num(size)} mm ({float(size) / wavelength_mm:.1f} wavelengths)"

    def _lateral_size_text(self, cfg) -> str:
        """'2.4 mm (18 A-lines)': mm and the A-lines the analysis will use (mm / lateral_res, as
        framework.get_seg_window_lens). A size given in A-lines is shown in mm with the mean A-line
        spacing inside the ROI (curvilinear arrays: the spacing grows with depth)."""
        size, units = cfg.lat_win_size, getattr(cfg, "lat_win_size_units", "mm")
        lat_res = getattr(self._image_data, "lateral_res", None)
        if units == "A-Lines":
            spacing = self._roi_line_spacing_mm()
            mm = f"{_num(size * spacing)} mm, mean in ROI" if spacing else "mm unknown"
            return f"{_num(size)} A-lines ({mm})"
        mm_size = size
        if units == "number of wavelength":
            try:
                mm_size = size * float(cfg.speed_of_sound) / float(cfg.center_frequency) * 1000
            except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                return _with_unit(_num(size), _unit_label(units))
        if not lat_res:
            return f"{_num(mm_size)} mm"
        lines = round(float(mm_size) / float(lat_res))
        text = f"{_num(mm_size)} mm ({lines} A-lines)"
        n_lines = self._image_data.rf_data.shape[-1]
        if lines > n_lines:
            text += f" - more than the {n_lines} A-lines in the image"
        return text

    def _update_display(self) -> None:
        """Update the display with current configuration data."""
        if not self._config_data:
            return
            
        cfg = self._config_data
        ui = self._ui

        # Values with their units after the number (same colour as the value); frequencies in MHz
        ui.transducerFreqLabel.setText("Transducer Frequency Band:")
        ui.analysisFreqLabel.setText("Analysis Frequency Band:")
        ui.centerFreqLabel.setText("Center Frequency:")
        ui.samplingFreqLabel.setText("Sampling Frequency:")
        ui.transducerFreqValue.setText(_with_unit(_mhz_band(cfg.transducer_freq_band), "MHz"))
        ui.analysisFreqValue.setText(_with_unit(_mhz_band(cfg.analysis_freq_band), "MHz"))
        ui.centerFreqValue.setText(_with_unit(_mhz(cfg.center_frequency), "MHz"))
        ui.samplingFreqValue.setText(_with_unit(_mhz(getattr(cfg, "sampling_frequency", None)), "MHz"))

        # Windowing parameters: sizes in the config's own units, coverage / overlaps in %
        ui.axWinSizeLabel.setText("Axial Window Size:")
        ui.latWinSizeLabel.setText("Lateral Window Size:")
        ui.windowThreshLabel.setText("Window Inclusion Coverage:")   # config.window_thresh
        ui.windowThreshLabel.setToolTip("A window is analysed only if more than this share of it lies inside the ROI.")
        ui.axialOverlapLabel.setText("Axial Overlap:")
        ui.lateralOverlapLabel.setText("Lateral Overlap:")
        ui.axWinSizeValue.setText(self._axial_size_text(cfg))
        ui.latWinSizeValue.setText(self._lateral_size_text(cfg))
        ui.windowThreshValue.setText(_with_unit(_num(cfg.window_thresh * 100), "%"))
        ui.axialOverlapValue.setText(_with_unit(_num(cfg.axial_overlap * 100), "%"))
        ui.lateralOverlapValue.setText(_with_unit(_num(cfg.lateral_overlap * 100), "%"))

        # Update 3D parameters
        if self._image_data.spatial_dims == 3:
            ui.corWinSizeLabel.setText("Coronal Window Size:")
            ui.coronalOverlapLabel.setText("Coronal Overlap:")
            ui.corWinSizeValue.setText(_with_unit(_num(cfg.cor_win_size), "mm"))
            ui.coronalOverlapValue.setText(_with_unit(_num(cfg.coronal_overlap * 100), "%"))
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


def _num(value) -> str:
    """Number without float noise: 95.0 -> '95', 2.2 -> '2.2', 0.9*100 -> '90'."""
    if value is None:
        return "Not set"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _mhz(hz) -> str:
    return "Not set" if hz is None else _num(float(hz) / 1e6)


def _mhz_band(band) -> str:
    try:
        lo, hi = band
    except (TypeError, ValueError):
        return "Not set"
    return f"{_mhz(lo)} - {_mhz(hi)}"


def _unit_label(units) -> str:
    """Window-size unit for a label: mm, wavelengths or A-lines."""
    return {"mm": "mm", "number of wavelength": "wavelengths", "A-Lines": "A-lines"}.get(units, units or "mm")


def _with_unit(text: str, unit: str) -> str:
    """'8.5 - 13.5' + 'MHz' -> '8.5 - 13.5 MHz'; '95' + '%' -> '95%'; 'Not set' stays as it is."""
    if text == "Not set" or not unit:
        return text
    return f"{text}%" if unit == "%" else f"{text} {unit}"
