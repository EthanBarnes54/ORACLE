"""The main window on the simulated rack, for the tests that drive it end to end.

The real window, a real acquisition process and the simulated meters,
writing under the test's own folder. The ELOG is a stand-in that keeps
every post (and checks an attachment is a real PNG before the post takes
it over, as the ELOG's worker does), and every dialog is kept instead of
shown, so a test can read what the operator would have been told.
"""

import multiprocessing
import os
import time

from cls_acquisition import settings
from cls_acquisition.acquisition import Rack_Link
from cls_acquisition.facility import Pv_Monitors
from cls_acquisition.hardware import Rack
from gui import main_window


def Pump(app, until, timeout=60.0):
    """Run the event loop until ``until()`` is true; False if it never was."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


class Recording_Elog():
    """The ELOG's interface, keeping each post as (kind, text, experiment, attachment)."""

    def __init__(self):
        self.Posts = []
        self.Enabled = False            # as on a simulated rack, which never posts
        self.Last_Error = None
        self.Attachments = []           # what each attachment held when it was posted

    def New_Entry(self, experiment):
        self.Posts.append(('new', '', experiment, None))

    def Append(self, text, experiment, attachment=None):
        if attachment is not None:
            with open(attachment, 'rb') as fh:
                self.Attachments.append(fh.read())
            os.remove(attachment)       # the post takes the file over, as the real one does
        self.Posts.append(('append', text, experiment, attachment))

    def Texts(self):
        return [text for kind, text, _, _ in self.Posts if kind == 'append']

    def Close(self, timeout=5.0):
        pass


def Open_Window(monkeypatch, tmp_path, warnings=(), **changes):
    """The window, its settings in ``tmp_path`` updated with ``changes``, set
    for a short single scan: 20 to 40 V in 2 V steps, two bunches a step
    every 10 ms. ``Dialogs`` keeps (title, text) of every message box."""
    monkeypatch.setattr(settings, 'APP_ROOT', str(tmp_path))
    # its own settings: a correction or settle model one window saves must not
    # be in force in the next test's
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(tmp_path / 'settings.yaml'))
    if changes:
        settings.Update(**changes)
    dialogs = []
    for kind in ('critical', 'warning', 'information'):
        monkeypatch.setattr(main_window.QMessageBox, kind,
                            lambda parent, title, text, *_, kind=kind, **__:
                            dialogs.append((title, text)))
    config = settings.Load()
    rack = Rack(config, multiprocessing.get_context('spawn'))
    link = Rack_Link(rack.Loop_Spec())
    link.Send('Ping', timeout=30)
    line, cooler = rack.Open_Meters()
    window = main_window.Main_Window(config, rack, link, line, cooler,
                                     Pv_Monitors({}, True), Recording_Elog(), warnings)
    window.Dialogs = dialogs
    scan = window.Scan
    scan.From_Boxes[0].setValue(20.0)
    scan.To_Boxes[0].setValue(40.0)
    scan.Step_Box.setValue(2.0)
    scan.Dwell_Box.setValue(10)
    scan.Per_Step_Box.setValue(2)
    scan.Type_Combo.setCurrentText('Single scan')
    return window


def Close_Window(app, window):
    """Stop whatever runs, close the window and check it left the process gone."""
    state = window._Link.State
    if window._Link.Alive() and state.Scanning.is_set():
        window._Link.Send('Stop_Now')
    Pump(app, lambda: window._Worker is None, timeout=30)
    window.close()
    assert not window._Link.Alive()


def Run_Files(window):
    import glob
    return sorted(glob.glob(os.path.join(window._Data_Dir, '*', '*', '*', 'run_*.asdf')))
