#!/usr/bin/env python3
"""ORACLE launcher.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Starts the ORACLE desktop application. The import sits under the main
guard because the acquisition process is spawned: it re-runs this file
as it starts, and must not load Qt.

Depends on: gui
"""

import sys

if __name__ == '__main__':
    from gui.main_window import main
    sys.exit(main())
