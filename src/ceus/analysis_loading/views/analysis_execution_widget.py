"""
Analysis Execution Widget for Analysis Loading

This widget displays the analysis summary, handles execution, and shows progress.
It allows users to review their configuration and execute the analysis.
"""

from typing import Optional, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QSizePolicy, QFileDialog)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...mvc.base_view import BaseViewMixin
from ..ui.analysis_execution_ui import Ui_analysisExecution
from engines.ceus.src.data_objs.image import UltrasoundImage
from engines.ceus.src.data_objs.seg import CeusSeg
from engines.ceus.src.time_series_analysis.curves.framework import CurvesAnalysis


class AnalysisExecutionWidget(QWidget, BaseViewMixin):
    """
    Widget for executing analysis and showing progress.
    
    This widget displays a summary of the selected analysis configuration,
    handles execution, shows progress, and displays results.
    """
    
    # Signals for communicating with controller
    execution_started = pyqtSignal(dict)  # execution_data
    analysis_confirmed = pyqtSignal(object)  # analysis_data (CurvesAnalysis)
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundImage, seg_data: CeusSeg, config_data, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisExecution()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Current state
        self._execution_summary: Dict = {}
        self._analysis_data: Optional[CurvesAnalysis] = None
        self._is_executing = False
        self._results_shown = False  # Track if results have been shown
        self._curve_quant = None  # CurveQuantifications result, stored for export
        
        # Progress simulation timer
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_progress_simulation)
        self._current_progress = 0
        
    def setup_ui(self) -> None:
        """Setup the user interface."""
        self._ui.setupUi(self)
        
        # Configure layout for execution
        self.setLayout(self._ui.full_screen_layout)
        
        # Configure stretch factors
        self._ui.full_screen_layout.setStretchFactor(self._ui.side_bar_layout, 1)
        self._ui.full_screen_layout.setStretchFactor(self._ui.analysis_execution_layout, 10)

        # Create a dedicated summary container inside the execution layout so we
        # can update the summary without destroying the progress UI and buttons
        # defined in the .ui file.
        self._summary_container = QWidget()
        self._summary_container.setStyleSheet("QWidget { background-color: transparent; }")
        self._summary_layout = QVBoxLayout(self._summary_container)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(6)
        # Insert the summary container just below the title label and above the
        # progress/status controls. The title label is at index 0 in the layout.
        try:
            self._ui.analysis_execution_layout.insertWidget(1, self._summary_container)
        except Exception:
            # Fallback to adding at the end if insertion index fails
            self._ui.analysis_execution_layout.addWidget(self._summary_container)

        # Update labels to reflect inputted image and phantom
        if self._image_data is not None:
            self._ui.image_path_input.setText(getattr(self._image_data, 'scan_name', "No image loaded"))
            self._ui.phantom_path_input.setText(getattr(self._image_data, 'phantom_name', ""))
        else:
            self._ui.image_path_input.setText("No image loaded")
            self._ui.phantom_path_input.setText("No phantom loaded")
            
        # Initially hide finish button
        self._ui.finish_button.setVisible(False)
        
    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self._ui.execute_button.clicked.connect(self._on_execute_clicked)
        self._ui.finish_button.clicked.connect(self._on_finish_clicked)
        self._ui.back_button.clicked.connect(self._on_back_clicked)
        
    def update_display(self, data) -> None:
        """Update the view with new data."""
        # This widget doesn't need to update with external data
        pass
        
    def set_execution_summary(self, execution_summary: Dict) -> None:
        """
        Set execution summary and update the display.

        Args:
            execution_summary: Dictionary containing execution summary data
        """
        self._execution_summary = execution_summary
        self._create_summary_display()

        # If there are no extra params the controller will auto-start; show
        # a "Running…" state immediately so the button doesn't flash.
        if not execution_summary.get('params'):
            self._ui.execute_button.setEnabled(False)
            self._ui.back_button.setEnabled(False)
            self._ui.progress_label.setText("Starting analysis...")
            self._is_executing = True
            self._current_progress = 0
            self._progress_timer.start(100)
        
    def _create_summary_display(self) -> None:
        """Create the summary display from execution data."""
        print(f"DEBUG: _create_summary_display called")
        print(f"DEBUG: _execution_summary = {self._execution_summary}")
        
        # Clear existing summary
        self._clear_summary_layout()
        
        layout = self._summary_layout
        
        # Analysis type
        analysis_type = self._execution_summary.get('analysis_type', 'Unknown')
        print(f"DEBUG: analysis_type = {analysis_type}")
        type_label = self._create_summary_item("Analysis Type:", analysis_type.title())
        layout.addWidget(type_label)
        
        # Selected functions
        functions = self._execution_summary.get('functions', [])
        functions_text = ', '.join([func.replace('_', ' ').title() for func in functions])
        if not functions_text:
            functions_text = "None selected"
        functions_label = self._create_summary_item("Selected Functions:", functions_text)
        layout.addWidget(functions_label)
        
        # Parameters summary
        params = self._execution_summary.get('params', {})
        if params:
            params_label = self._create_summary_item("Parameters:", f"{len(params)} parameters configured")
            layout.addWidget(params_label)
            
            # Show key parameters
            for param_name, param_value in list(params.items())[:5]:  # Show first 5 params
                formatted_name = param_name.replace('_', ' ').title()
                if isinstance(param_value, dict):
                    value_text = f"Complex parameter ({len(param_value)} settings)"
                elif isinstance(param_value, (int, float)):
                    value_text = f"{param_value}"
                else:
                    value_text = str(param_value)[:50] + ("..." if len(str(param_value)) > 50 else "")
                    
                param_label = self._create_summary_item(f"  {formatted_name}:", value_text, is_sub_item=True)
                layout.addWidget(param_label)
                
            if len(params) > 5:
                more_label = self._create_summary_item("", f"... and {len(params) - 5} more parameters", is_sub_item=True)
                layout.addWidget(more_label)
        else:
            params_label = self._create_summary_item("Parameters:", "No additional parameters")
            layout.addWidget(params_label)
        
        # Ensure the summary has stretch at the end to push items up
        layout.addStretch()
        
    def _create_summary_item(self, label_text: str, value_text: str, is_sub_item: bool = False) -> QWidget:
        """
        Create a summary item widget.
        
        Args:
            label_text: Label text
            value_text: Value text
            is_sub_item: Whether this is a sub-item (indented)
            
        Returns:
            QWidget containing the summary item
        """
        container = QWidget()
        container.setStyleSheet("QWidget { background-color: transparent; }")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10 if is_sub_item else 0, 5, 0, 5)
        
        # Label
        label = QLabel(label_text)
        label.setStyleSheet(f"""
            QLabel {{
                color: rgb({'180, 180, 180' if is_sub_item else '220, 220, 220'});
                font-size: {'10px' if is_sub_item else '11px'};
                font-weight: {'normal' if is_sub_item else 'bold'};
                background-color: transparent;
            }}
        """)
        label.setMinimumWidth(150)
        
        # Value
        value = QLabel(value_text)
        value.setStyleSheet(f"""
            QLabel {{
                color: rgb({'160, 160, 160' if is_sub_item else '255, 255, 255'});
                font-size: {'10px' if is_sub_item else '11px'};
                background-color: transparent;
            }}
        """)
        value.setWordWrap(True)
        
        layout.addWidget(label)
        layout.addWidget(value, 1)
        
        container.setLayout(layout)
        return container
        
    def _clear_summary_layout(self) -> None:
        """Clear all widgets from the summary container only."""
        if hasattr(self, '_summary_layout') and self._summary_layout is not None:
            while self._summary_layout.count():
                child = self._summary_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                
    def show_results(self, analysis_data: CurvesAnalysis) -> None:
        """
        Show analysis results. If TIC data is present, embed a matplotlib plot
        with the raw TIC curve and a Gaussian fit, plus derived parameters.

        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        self._results_shown = False

        # Update progress bar
        self._ui.progress_bar.setValue(100)
        self._ui.progress_label.setText("Analysis completed successfully!")

        # Swap buttons
        self._ui.execute_button.setVisible(False)
        self._ui.finish_button.setVisible(True)
        self._ui.finish_button.setEnabled(True)
        self._ui.finish_button.setText("Export / Finish")
        self._ui.back_button.setEnabled(True)
        self._progress_timer.stop()
        self._is_executing = False

        # Run curve quantification (lognormal fit) via the engine entrypoint
        try:
            curves_dict = analysis_data.curves[0] if analysis_data.curves else {}
            tic_key = next((k for k in curves_dict if k.upper() == 'TIC'), None)
            if tic_key is not None:
                from engines.ceus.src.entrypoints import curve_quantification_step
                curve_quant = curve_quantification_step(
                    analysis_data,
                    ['lognormal_fit_full'],
                    None,  # no file save yet — user chooses path on Export
                    curves_to_fit=[tic_key],
                )
                self._curve_quant = curve_quant  # store for export
                self._show_tic_plot(
                    np.array(analysis_data.time_arr, dtype=float),
                    np.array(curves_dict[tic_key], dtype=float),
                    curve_quant.data_dict[0],
                    tic_key,
                )
        except Exception as e:
            print(f"DEBUG: Could not run quantification / render TIC plot: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # TIC plot with log-normal fit overlay
    # ------------------------------------------------------------------

    def _show_tic_plot(self, time_arr: np.ndarray, tic_arr: np.ndarray,
                       quant_dict: dict, tic_key: str) -> None:
        """Embed a matplotlib figure showing TIC + log-normal fit + parameters."""
        from engines.ceus.src.curve_quantification.transforms import bolus_lognormal

        self._clear_summary_layout()
        layout = self._summary_layout

        # Pull fitted params from the quantification result dict
        prefix = f'_full_{tic_key}'
        auc   = quant_dict.get(f'AUC{prefix}',   np.nan)
        pe    = quant_dict.get(f'PE{prefix}',    np.nan)
        tp    = quant_dict.get(f'TP{prefix}',    np.nan)
        mtt   = quant_dict.get(f'MTT{prefix}',   np.nan)
        t0    = quant_dict.get(f'T0{prefix}',    np.nan)
        mu    = quant_dict.get(f'Mu{prefix}',    np.nan)
        sigma = quant_dict.get(f'Sigma{prefix}', np.nan)
        fit_ok = not np.isnan(auc)

        if fit_ok:
            t_fine = np.linspace(time_arr[0], time_arr[-1], 500)
            normalizer = float(np.max(tic_arr) - np.min(tic_arr)) or 1.0
            min_val    = float(np.min(tic_arr))
            tic_fit = bolus_lognormal(t_fine, auc / normalizer, mu, sigma, t0) * normalizer + min_val

        # ---- Matplotlib canvas ----
        fig = Figure(figsize=(6, 3.5), facecolor='#2a2a2a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1a1a1a')
        ax.plot(time_arr, tic_arr, color='#4fc3f7', linewidth=1.5, label='TIC (raw)')
        if fit_ok:
            ax.plot(t_fine, tic_fit, color='#ff8a65', linewidth=2.0, linestyle='--',
                    label='Log-normal fit')
        ax.set_xlabel('Time', color='white', fontsize=9)
        ax.set_ylabel('Mean Intensity', color='white', fontsize=9)
        ax.set_title('Time Intensity Curve', color='white', fontsize=11)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        ax.legend(fontsize=8, facecolor='#333333', labelcolor='white', framealpha=0.8)
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas)

        # ---- Parameter table ----
        params_widget = QWidget()
        params_widget.setStyleSheet("background-color: transparent;")
        params_layout = QHBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 4, 0, 0)
        params_layout.setSpacing(24)

        param_items = [
            ("Peak Enhancement (PE)", f"{pe:.4f}"  if fit_ok else "N/A"),
            ("Time to Peak (TP)",     f"{tp:.2f}"  if fit_ok else "N/A"),
            ("Mean Transit Time (MTT)", f"{mtt:.2f}" if fit_ok else "N/A"),
            ("Area Under Curve (AUC)", f"{auc:.4f}" if fit_ok else "N/A"),
            ("Arrival Time (T0)",     f"{t0:.2f}"  if fit_ok else "N/A"),
            ("μ",                     f"{mu:.3f}"  if fit_ok else "N/A"),
            ("σ",                     f"{sigma:.3f}" if fit_ok else "N/A"),
        ]
        for name, value in param_items:
            item = QWidget()
            item.setStyleSheet("background-color: transparent;")
            vl = QVBoxLayout(item)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(2)
            lbl = QLabel(name)
            lbl.setStyleSheet("color: rgb(170,170,170); font-size: 10px;")
            val = QLabel(value)
            val.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
            vl.addWidget(lbl)
            vl.addWidget(val)
            params_layout.addWidget(item)

        params_layout.addStretch()
        layout.addWidget(params_widget)
        
    def _on_execute_clicked(self) -> None:
        """Handle execute button click."""
        print(f"DEBUG: Execute button clicked!")
        print(f"DEBUG: _is_executing = {self._is_executing}")
        print(f"DEBUG: execute_button enabled = {self._ui.execute_button.isEnabled()}")
        print(f"DEBUG: execute_button visible = {self._ui.execute_button.isVisible()}")
        
        if not self._is_executing:
            print(f"DEBUG: Starting analysis execution...")
            # Start execution
            self._is_executing = True
            self._ui.execute_button.setEnabled(False)
            self._ui.back_button.setEnabled(False)
            
            # Reset progress
            self._current_progress = 0
            self._ui.progress_bar.setValue(0)
            self._ui.progress_label.setText("Starting analysis...")
            
            # Start progress simulation
            self._progress_timer.start(100)  # Update every 100ms
            
            # Emit execution signal
            execution_data = {
                'summary': self._execution_summary,
                'timestamp': self._get_current_timestamp()
            }
            print(f"DEBUG: Emitting execution_started signal with data: {execution_data}")
            self.execution_started.emit(execution_data)
        else:
            print(f"DEBUG: Analysis already executing, ignoring click")

    def _on_finish_clicked(self) -> None:
        """Open a save dialog to export quantification results as CSV, then confirm."""
        if hasattr(self, '_curve_quant') and self._curve_quant is not None:
            scan_name = getattr(self._analysis_data.image_data, 'scan_name', 'tic_results') \
                        if self._analysis_data else 'tic_results'
            default_name = f"{scan_name}_fit_params.csv"
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Fit Parameters", default_name,
                "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                if not path.endswith('.csv'):
                    path += '.csv'
                import pandas as pd
                # Export fit parameters
                pd.DataFrame(self._curve_quant.data_dict).to_csv(path, index=False)
                print(f"Fit parameters exported to {path}")
                # Export raw TIC curve alongside fit params
                if self._analysis_data and self._analysis_data.curves:
                    curves_dict = self._analysis_data.curves[0]
                    tic_key = next((k for k in curves_dict if k.upper() == 'TIC'), None)
                    if tic_key is not None:
                        tic_path = path.replace('.csv', '_tic_curve.csv')
                        tic_df = pd.DataFrame({
                            'Time': self._analysis_data.time_arr,
                            'Intensity': curves_dict[tic_key],
                        })
                        tic_df.to_csv(tic_path, index=False)
                        print(f"TIC curve exported to {tic_path}")
        self._on_continue_to_visualization()

    def _on_continue_to_visualization(self) -> None:
        """Handle continue to visualization button click."""
        print(f"DEBUG: Continue to visualization button clicked!")
        if self._analysis_data:
            print(f"DEBUG: Emitting analysis_confirmed signal with analysis data")
            self.analysis_confirmed.emit(self._analysis_data)
        else:
            print(f"DEBUG: No analysis data available")

    def _show_analysis_results_display(self) -> None:
        """Show detailed analysis results display."""
        # Clear existing summary
        self._clear_summary_layout()
        
        # Create results display
        layout = self._summary_layout
        
        # Add results header
        header_label = QLabel("Analysis Results")
        header_label.setStyleSheet("""
            QLabel {
                color: rgb(0, 255, 0);
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header_label)
        
        # Add analysis data summary
        if self._analysis_data:
            # Show basic analysis info
            analysis_info = self._create_summary_item("Analysis Status:", "Completed Successfully")
            layout.addWidget(analysis_info)
            
            # Show functions that were executed
            functions_text = ', '.join([func.replace('_', ' ').title() for func in self._execution_summary.get('functions', [])])
            functions_info = self._create_summary_item("Executed Functions:", functions_text)
            layout.addWidget(functions_info)
            
            # Add placeholder for results summary
            results_info = self._create_summary_item("Results:", "Analysis data ready for visualization")
            layout.addWidget(results_info)
        else:
            error_info = self._create_summary_item("Error:", "No analysis data available")
            layout.addWidget(error_info)
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        if not self._is_executing:
            self.back_requested.emit()
            
    def _update_progress_simulation(self) -> None:
        """Update progress bar simulation during analysis."""
        if self._current_progress < 95:  # Don't go to 100% until analysis is actually done
            # Simulate progress with varying speed
            if self._current_progress < 30:
                increment = 2  # Fast start
            elif self._current_progress < 70:
                increment = 1  # Medium progress
            else:
                increment = 0.5  # Slow near end
                
            self._current_progress += increment
            self._ui.progress_bar.setValue(int(self._current_progress))
            
            # Update status messages
            if self._current_progress < 20:
                self._ui.progress_label.setText("Initializing analysis...")
            elif self._current_progress < 40:
                self._ui.progress_label.setText("Processing windows...")
            elif self._current_progress < 70:
                self._ui.progress_label.setText("Computing parameters...")
            elif self._current_progress < 90:
                self._ui.progress_label.setText("Finalizing results...")
            else:
                self._ui.progress_label.setText("Almost complete...")
                
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def show_error(self, error_message: str) -> None:
        """
        Override to show error and reset state.
        
        Args:
            error_message: Error message to display
        """
        super().show_error(error_message)
        
        # Reset execution state
        self._is_executing = False
        self._progress_timer.stop()
        
        # Reset UI state
        self._ui.execute_button.setEnabled(True)
        self._ui.back_button.setEnabled(True)
        self._ui.progress_bar.setValue(0)
        self._ui.progress_label.setText("Ready to execute analysis")
