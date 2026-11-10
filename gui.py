#!/usr/bin/env python3
"""ORACLE launcher.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The import sits under the main guard: the spawned acquisition process
re-runs this file and must not load Qt.

Depends on: gui
"""

import sys

if __name__ == '__main__':
    from gui.main_window import main
    sys.exit(main())
