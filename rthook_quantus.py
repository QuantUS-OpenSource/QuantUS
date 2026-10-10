import sys
import os

if hasattr(sys, '_MEIPASS'):
    bad = [p for p in sys.path
           if 'engines' in p.replace('\\', '/').lower()
           or '__editable__' in p.lower()]
    for p in bad:
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    if sys._MEIPASS in sys.path:
        sys.path.remove(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)