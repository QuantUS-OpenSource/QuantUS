# Visualization Loading Module

This module provides the GUI components for selecting visualization types and functions based on the analysis results. It follows the MVC architecture pattern used throughout the QuantUS GUI.

## Overview

The visualization loading module allows users to:
1. Select visualization types (e.g., paramap)
2. Choose visualization functions based on the analysis results
3. Configure visualization parameters
4. Preview the visualization configuration
5. Execute the visualization

## Architecture

The module follows the MVC (Model-View-Controller) pattern:

### Controller
- `VisualizationLoadingController`: Manages the visualization loading process and coordinates between the model and views

### Views
- `VisualizationLoadingViewCoordinator`: Main view coordinator that manages all sub-views
- `VisualizationTypeSelectionWidget`: Widget for selecting visualization types
- `VisualizationFunctionSelectionWidget`: Widget for selecting visualization functions with recommendations
- `VisualizationParamsWidget`: Widget for configuring visualization parameters
- `VisualizationPreviewWidget`: Widget for previewing the visualization configuration

## Analysis to Visualization Mapping

The module automatically recommends visualization functions based on the analysis functions that were executed:

| Analysis Function | Recommended Visualization Functions |
|------------------|-------------------------------------|
| `compute_power_spectra` | `paramaps` |
| `lizzi_feleppa` | `paramaps` |
| `attenuation_coef` | `paramaps` |
| `bsc` | `paramaps` |
| `nakagami_params` | `paramaps` |
| `hscan` | `plot_hscan_result`, `plot_hscan_wavelets` |
| `central_freq_shift` | `paramaps` |

## Usage

### Basic Usage

```python
from src.qus.visualization_loading import VisualizationLoadingController
from engines.qus.quantus.analysis.paramap.framework import ParamapAnalysis

# Create controller with model and analysis data
controller = VisualizationLoadingController(model, analysis_data)

# Get the main widget
widget = controller.get_widget()

# Show the widget
widget.show()
```

### Integration with Application Controller

The visualization loading module is designed to integrate with the main application controller:

```python
# In application controller
def _initialize_visualization_loading(self, analysis_data):
    self._visualization_loading_controller = VisualizationLoadingController(
        self._model, analysis_data
    )
    
    widget = self._visualization_loading_controller.get_widget()
    self._main_window.setCentralWidget(widget)
    
    # Connect signals
    self._visualization_loading_controller.user_action.connect(self._on_visualization_action)
    self._visualization_loading_controller.back_requested.connect(self._navigate_to_analysis_loading)
```

## Features

### Automatic Recommendations
The module automatically suggests appropriate visualization functions based on the analysis results, making it easier for users to select relevant visualizations.

### Parameter Configuration
Users can configure visualization parameters through a dynamic form that adapts to the selected visualization type and functions.

### Preview
A preview panel shows the current visualization configuration, including selected type, functions, and parameters.

### Validation
The module validates the visualization configuration before execution to ensure all required components are selected.

## File Structure

```
quantus/gui/visualization_loading/
├── __init__.py
├── README.md
├── visualization_loading_controller.py
├── visualization_loading_view_coordinator.py
├── ui/
│   ├── visualization_type_selection.ui
│   ├── visualization_function_selection.ui
│   ├── visualization_params.ui
│   └── visualization_preview.ui
└── views/
    ├── __init__.py
    ├── visualization_type_selection_widget.py
    ├── visualization_function_selection_widget.py
    ├── visualization_params_widget.py
    └── visualization_preview_widget.py
```

## Testing

Run the test script to verify the module works correctly:

```bash
python test_visualization_loading.py
```

This will open a GUI window showing the visualization loading interface with mock data.
