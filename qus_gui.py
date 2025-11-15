"""
Main entry point for QuantUS GUI using MVC architecture

This replaces the original main.py with a clean MVC-based implementation
that provides better separation of concerns and maintainability.
"""

from src.qus.application_controller import create_application


def main():
    """Main entry point for the QuantUS GUI application."""
    try:
        # Create the application
        app_controller = create_application()
        
        # Run the application
        app_controller.run()
        
    except Exception as e:
        print(f"Error starting QuantUS application: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
