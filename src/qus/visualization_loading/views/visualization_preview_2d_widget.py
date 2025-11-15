"""
Visualization Function Selection Widget for QuantUS GUI

This widget allows users to select visualization functions with recommendations
based on the analysis results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, Qt, QSize

from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis
from engines.qus.quantus.data_objs.analysis_config import RfAnalysisConfig
from engines.qus.quantus.data_objs.image import UltrasoundRfImage
from engines.qus.quantus.data_objs.seg import BmodeSeg
from engines.qus.quantus.data_objs.visualizations import ParamapDrawingBase
from ...mvc.base_view import BaseViewMixin
from ..ui.visualization_preview_2d_ui import Ui_visualization_preview_2d


class VisualizationPreview2DWidget(QWidget, BaseViewMixin):
    """Widget for displaying 2D visualization previews."""
    # Signals for communicating with controller
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_visualization_preview_2d()
        self._image_data = image_data
        self._visualization_folder: Optional[Path] = None

        # Setup UI
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self._ui.setupUi(self)

        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.preview_layout, 10)

        # Update image and phantom paths
        self._ui.image_path_input.setText(self._image_data.scan_name)
        self._ui.phantom_path_input.setText(self._image_data.phantom_name)

        self._setup_matplotlib_canvas()

    def _setup_matplotlib_canvas(self) -> None:
        """Setup matplotlib canvas for segmentation display."""
        # Create matplotlib figure and canvas
        fig, ax = plt.subplots(figsize=(8, 6))
        self._matplotlib_canvas = FigureCanvas(fig)
        self._matplotlib_canvas.figure.patch.set_facecolor((0, 0, 0, 0))
        ax.axis('off')

        # Add canvas to the preview frame widget
        layout = QHBoxLayout(self._ui.visualization_display_frame)
        layout.addWidget(self._matplotlib_canvas)
        self._ui.visualization_display_frame.setLayout(layout)
        self._matplotlib_canvas.draw()

    def set_visualization_folder(self, visualization_folder: str) -> None:
        """Set the folder containing visualization results and populate the dropdown."""
        self._visualization_folder = Path(visualization_folder)

        # Iterate through the results data
        self._ui.visualization_dropdown.addItem("")
        for file in self._visualization_folder.glob("*.png"):
            if file.name.endswith("_legend.png"):
                continue
            self._ui.visualization_dropdown.addItem(file.stem)

    def _connect_signals(self):
        """Connect signals."""
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        self._ui.visualization_dropdown.currentTextChanged.connect(self._on_dropdown_changed)

    def _on_dropdown_changed(self, text: str) -> None:
        """Handle dropdown selection change."""
        fig = self._matplotlib_canvas.figure
        fig.clear()
        
        if not text:
            self._matplotlib_canvas.draw()
            return
        
        if text.endswith("_paramap"):
            legend_path = self._visualization_folder / f"{text[:-len('_paramap')]}_legend.png"
            legend_im = plt.imread(legend_path)
            ax_legend = fig.add_axes([0.8, 0.1, 0.1, 0.8])
            ax_legend.imshow(legend_im)
            ax_legend.axis('off')
            ax_legend.set_position([0.8, 0.1, 0.1, 0.8])
        
        image_path = self._visualization_folder / f"{text}.png"
        im = plt.imread(image_path)  # For testing if the image can be read
        
        ax = self._matplotlib_canvas.figure.add_subplot(111)
        if text.endswith("_paramap"):
            ax.set_position([0, 0, 0.8, 0.8])
        else:
            ax.set_position([0, 0, 1, 1])
        ax.imshow(im)
        ax.axis('off')
        self._matplotlib_canvas.draw()

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
