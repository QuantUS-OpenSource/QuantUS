"""
Analysis Loading Views Module

This module contains the individual widget views for the analysis loading workflow.
"""

from .analysis_function_selection_widget import AnalysisFunctionSelectionWidget
from .analysis_execution_widget import AnalysisExecutionWidget

__all__ = [
    'AnalysisFunctionSelectionWidget', 
    'AnalysisExecutionWidget'
]
