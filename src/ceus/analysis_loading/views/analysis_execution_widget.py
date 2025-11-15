"""
Analysis Execution Widget for Analysis Loading

This widget displays the analysis summary, handles execution, and shows progress.
It allows users to review their configuration and execute the analysis.
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont

from quantus.gui.mvc.base_view import BaseViewMixin
from quantus.gui.analysis_loading.ui.analysis_execution_ui import Ui_analysisExecution
from quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from quantus.analysis.paramap.framework import ParamapAnalysis


class AnalysisExecutionWidget(QWidget, BaseViewMixin):
    """
    Widget for executing analysis and showing progress.
    
    This widget displays a summary of the selected analysis configuration,
    handles execution, shows progress, and displays results.
    """
    
    # Signals for communicating with controller
    execution_started = pyqtSignal(dict)  # execution_data
    analysis_confirmed = pyqtSignal(object)  # analysis_data (ParamapAnalysis)
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, seg_data: BmodeSeg, config_data: RfAnalysisConfig, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent)
        self.__init_base_view__(parent)
        self._ui = Ui_analysisExecution()
        self._image_data = image_data
        self._seg_data = seg_data
        self._config_data = config_data
        
        # Current state
        self._execution_summary: Dict = {}
        self._analysis_data: Optional[ParamapAnalysis] = None
        self._is_executing = False
        self._results_shown = False  # Track if results have been shown
        
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
            self._ui.image_path_input.setText(self._image_data.scan_name or "No image loaded")
            self._ui.phantom_path_input.setText(self._image_data.phantom_name or "No phantom loaded")
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
        print(f"DEBUG: set_execution_summary called with execution_summary = {execution_summary}")
        self._execution_summary = execution_summary
        print(f"DEBUG: Calling _create_summary_display...")
        self._create_summary_display()
        print(f"DEBUG: set_execution_summary completed")
        
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
                
    def show_results(self, analysis_data: ParamapAnalysis) -> None:
        """
        Show analysis results.
        
        Args:
            analysis_data: Completed analysis data
        """
        self._analysis_data = analysis_data
        self._results_shown = False  # Reset flag for new results
        
        # Update progress
        self._ui.progress_bar.setValue(100)
        self._ui.progress_label.setText("Analysis completed successfully!")
        
        # Show finish button, hide execute button
        self._ui.execute_button.setVisible(False)
        self._ui.finish_button.setVisible(True)
        self._ui.finish_button.setEnabled(True)
        try:
            # Make the intent explicit in the UI
            self._ui.finish_button.setText("Continue to Visualization")
        except Exception:
            # If text property isn't available for some reason, ignore
            pass
        
        # Enable back button
        self._ui.back_button.setEnabled(True)
        
        # Stop any progress simulation
        self._progress_timer.stop()
        self._is_executing = False
        
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
        """Handle finish button click."""
        # Go directly to visualization without the intermediate step
        print(f"DEBUG: Finish button clicked - going directly to visualization")
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
