"""
Segmentation Preview Widget for Segmentation Loading
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QWidget, QHBoxLayout
from PyQt6.QtCore import pyqtSignal

from src.qus.mvc.base_view import BaseViewMixin
from src.qus.seg_loading.ui.roi_preview_ui import Ui_constructRoi
from engines.qus.quantus.data_objs import UltrasoundRfImage, BmodeSeg


class RoiPreviewWidget(QWidget, BaseViewMixin):
    """
    Widget for previewing and confirming segmentation.
    
    This is the final step in the segmentation loading process where users
    can preview the loaded segmentation and confirm it before proceeding.
    Designed to be used within the main application widget stack.
    """
    
    # Signals for communicating with controller
    segmentation_confirmed = pyqtSignal(object)  # BmodeSeg data
    back_requested = pyqtSignal()
    close_requested = pyqtSignal()
    
    def __init__(self, seg_data: BmodeSeg, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_constructRoi()
        self._seg_data = seg_data
        self._image_data = image_data
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._frame = seg_data.frame
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for segmentation preview only - use the main layout
        self.setLayout(self._ui.main_layout)
        
        # Configure stretch factors for confirmation
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.confirmation_layout, 10)
        
        # Ensure the layout fills the entire widget
        self._ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.main_layout.setSpacing(0)
        self._ui.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self._ui.full_screen_layout.setSpacing(0)
        
        # Update labels to reflect inputted image, phantom, and frame
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)
        self._ui.segmentation_name_val.setText(self._seg_data.seg_name)

        if self._image_data.rf_data.ndim == 2:
            self._ui.confirmation_frame_label.hide()
        else:
            self._ui.confirmation_frame_label.setText(f"Frame: {self._frame}")

        # Setup matplotlib canvas for segmentation preview
        self._setup_matplotlib_canvas()
        
        # Display segmentation preview
        self._display_segmentation_preview()
        
    def _setup_matplotlib_canvas(self) -> None:
        """Setup matplotlib canvas for segmentation display."""
        # Create matplotlib figure and canvas
        fig, ax = plt.subplots(figsize=(8, 6))
        self._matplotlib_canvas = FigureCanvas(fig)
        self._matplotlib_canvas.figure.patch.set_facecolor((0, 0, 0, 0))
        
        # Add canvas to the preview frame widget
        layout = QHBoxLayout(self._ui.im_display_frame)
        layout.addWidget(self._matplotlib_canvas)
        self._ui.im_display_frame.setLayout(layout)
    
    def _connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.confirm_seg_button.clicked.connect(self._on_confirm_segmentation)
        self._ui.back_from_confirm_button.clicked.connect(self._on_back_clicked)
            
    def _display_segmentation_preview(self) -> None:
        """Display preview of the segmentation."""
        if not self._matplotlib_canvas:
            return
            
        try:
            fig = self._matplotlib_canvas.figure
            fig.clear()
            self._ax = fig.add_subplot(111)
            self._ax.set_position([0, 0, 1, 1])

            im = self._image_data.sc_bmode if self._image_data.sc_bmode is not None else self._image_data.bmode
            if im.ndim == 2:
                self._displayed_im = im
            else:
                if self._frame < 0 or self._frame >= im.shape[0]:
                    raise ValueError(f"Frame {self._frame} is out of bounds for image with {im.shape[0]} frames")
                self._displayed_im = im[self._frame]

            self._plot_im_on_ax(self._ax)

            # Plot segmentation data on the same axes
            splines = hasattr(self._seg_data, 'sc_splines') and self._seg_data.sc_splines or self._seg_data.splines
            x, y = splines[0], splines[1]
            self._ax.plot(x, y, color='cyan', linewidth=1)
            
            self._matplotlib_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying image: {e}")
            
    def _plot_im_on_ax(self, ax) -> None:
        """Plot the image on the given axes."""
        if self._image_data.sc_bmode is not None:
            width = self._displayed_im.shape[1]*self._image_data.sc_lateral_res
            height = self._displayed_im.shape[0]*self._image_data.sc_axial_res
        else:
            width = self._displayed_im.shape[1]*self._image_data.lateral_res
            height = self._displayed_im.shape[0]*self._image_data.axial_res
        aspect = width/height
        im = ax.imshow(self._displayed_im, cmap="gray")
        extent = im.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        ax.axis("off")

    def _on_confirm_segmentation(self) -> None:
        """Handle segmentation confirmation."""
        if not self._seg_data:
            self.show_error("No segmentation data to confirm")
            return
            
        self.segmentation_confirmed.emit(self._seg_data)
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
        
    def get_segmentation_data(self) -> BmodeSeg:
        """Get the current segmentation data."""
        return self._seg_data
