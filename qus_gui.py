"""
Main entry point for QuantUS QUS GUI.
Handles frozen (PyInstaller) and dev contexts.
"""
import sys
import os


def _configure_frozen_env() -> None:
    """One-time setup when running as a PyInstaller bundle."""
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return

    # Make sure Qt can find its platform plugin (e.g. qwindows.dll / libqxcb.so)
    qt_plugin_path = os.path.join(sys._MEIPASS, "PyQt6", "Qt6", "plugins")
    if os.path.isdir(qt_plugin_path):
        os.environ.setdefault("QT_PLUGIN_PATH", qt_plugin_path)

    # Suppress Qt's attempt to load system OpenGL on Linux
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_egl")


def main() -> int:
    _configure_frozen_env()

    try:
        from src.qus.application_controller import create_application
        app_controller = create_application()
        app_controller.run()
    except Exception as e:
        # In frozen mode there's no terminal, so show a dialog
        if getattr(sys, "frozen", False):
            try:
                from PyQt6.QtWidgets import QApplication, QMessageBox
                _app = QApplication.instance() or QApplication(sys.argv)
                QMessageBox.critical(None, "QuantUS Error", str(e))
            except Exception:
                pass
        else:
            print(f"Error starting QuantUS: {e}")
            raise
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())