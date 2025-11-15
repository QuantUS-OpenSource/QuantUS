# QuantUS GUI Module for QUS

This directory contains the Qt-based graphical user interface for QuantUS, built using a Model-View-Controller (MVC) architecture with specialized coordinators for managing complex workflows. The GUI provides an intuitive interface for ultrasound tissue characterization analysis while maintaining clean separation between UI logic and backend functionality.

## Architecture Overview

The QuantUS GUI follows a modified MVC pattern with additional coordinator components to manage complex multi-step workflows:

```
src/qus/
├── application_controller.py  # Main application controller
├── application_model.py       # Unified application model
├── mvc/                       # Base MVC framework
│   ├── base_controller.py     # Base controller class
│   ├── base_model.py         # Base model class  
│   └── base_view.py          # Base view mixin
├── image_loading/            # Image loading module
│   ├── image_loading_controller.py
│   ├── image_loading_view_coordinator.py
│   ├── ui/                   # Qt Designer .ui files
│   └── views/               # Individual view widgets
├── seg_loading/             # Segmentation loading module
│   ├── seg_loading_controller.py
│   ├── seg_loading_view_coordinator.py
│   ├── ui/                  # Qt Designer .ui files
│   └── views/              # Individual view widgets
└── saveQt.sh               # Script to convert .ui to .py files
```

## MVC Architecture Components

### Model (`application_model.py`)

Models handle all data management and business logic, serving as the bridge between the GUI and the QuantUS backend. This model contains:
- **Unified Data Store**: Central repository for all application data coming from the backend and the frontend
- **Backend Integration**: Interfaces with QuantUS core modules (`entrypoints.py`, plugin systems)
- **Asynchronous Operations**: Uses QThread workers for time-consuming operations

### Controllers (`*_controller.py`)

Controllers coordinate between model and views, handling user actions and updating views based on model changes:

#### **ApplicationController** 
- **Screen Navigation**: Manages QStackedWidget for different application screens
- **Workflow Coordination**: Orchestrates multi-step processes (image loading → segmentation → analysis)
- **Model-View Binding**: Connects model signals to view updates

#### **Specialized Controllers**
Each major section of the analysis pipeline has its own controller. The purpose here is to manage the I/O for this section and to navigate through different stages of each section. Note these controllers interact with the views indirectly through coordinators, and don't interact directly with the views themselves.
- **ImageLoadingController**: Manages ultrasound image loading workflow
- **SegmentationLoadingController**: Handles ROI/segmentation definition

### Views (`*_view.py`, `views/`)

Views handle UI presentation and user interactions using Qt widgets:

#### **Base View Pattern**
All views inherit from `BaseViewMixin` which provides:
- Consistent styling matching QuantUS theme
- Standard signal patterns for MVC communication
- Loading state management
- Error display capabilities

#### **Qt Designer Integration**
Views use Qt Designer `.ui` files converted to Python:

1. **Design in Qt Designer**: Create `.ui` files with visual layouts
2. **Convert to Python**: Use `./saveQt.sh` to generate `*_ui.py` files  
3. **Import in Views**: Views import and use the generated UI classes


## View Coordinators

Coordinators are a specialized pattern for managing complex multi-step workflows within a single functional area. Individual `.ui` files should only contain a single menu each, and these menus can be strung together using coordinators.

### **Purpose of Coordinators**
- **Multi-Step Workflows**: Manage sequences like scan type selection → file selection → loading
- **State Management**: Track current step, validate transitions, maintain workflow state
- **View Composition**: Combine multiple specialized widgets into cohesive workflows
- **Controller Interface**: Provide unified interface for controllers to interact with complex views

### **File Naming Convention**
- **UI Files**: `widget_name.ui` (designed in Qt Designer)
- **Generated Python**: `widget_name_ui.py` (auto-generated, do not edit manually)
- **View Classes**: `widget_name_widget.py` (hand-written, imports generated UI)

Example:
```
scan_type.ui              # Qt Designer file
scan_type_ui.py          # Generated Python UI class  
scan_type_widget.py      # Hand-written view class
```

## Signal-Slot Communication

The MVC architecture uses Qt's signal-slot mechanism for loose coupling. The controllers have information fed to them from model and views via signals. The controllers pass information to the model via function calls and they pass information to the views via class initializations and function calls. Note the views and the model never directly interact with each other.

## Development Guidelines

### **Adding New Functionality**

1. **Design UI in Qt Designer**:
   - Create `.ui` file with visual layout
   - Follow QuantUS styling conventions
   - Use descriptive widget names

2. **Run UI Conversion**:
   ```bash
   ./saveQt.sh
   ```

3. **Create View Class**:
   ```python
   class NewWidget(QWidget, BaseViewMixin):
       def __init__(self):
           super().__init__()
           self._ui = Ui_GeneratedClass()
           self._ui.setupUi(self)
   ```
4. **Create/Extend Controller**:
   ```python
   # Handle user actions and coordinate model-view
   def handle_new_action(self, action_data):
       # Process user action
       # Update model
       # Update view
   ```

### **Best Practices**

1. **Separation of Concerns**:
   - Views: Only UI logic and user interaction
   - Models: Only data and business logic
   - Controllers: Only coordination between model and view

2. **Signal-Slot Communication**:
   - Use signals for loose coupling
   - Connect signals in controllers
   - Emit signals for state changes

3. **Backend Integration**:
   - Access backend only through models
   - Use QuantUS entry points for processing
   - Handle backend errors gracefully

4. **Threading**:
   - Use QThread for time-consuming operations
   - Never access UI from worker threads
   - Use signals to communicate with main thread

5. **UI Consistency**:
   - Follow existing QuantUS styling
   - Use `BaseViewMixin` for standard patterns
   - Test with different screen sizes
