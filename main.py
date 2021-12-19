#!/usr/bin/env python3
"""Main asteroid-automator script."""

import platform

PLATFORM_SYS = platform.system()

if PLATFORM_SYS == "Windows":
    import d3dshot
elif PLATFORM_SYS == "Linux":
    import pyautogui


class AutomatableGame:
    """Base class for each platform."""

    def __init__(self):
        self.loc = pyautogui.locateOnScreen("images/app.png")

    def fetch_sshot(self):
        """Creates a screenshot, and returns it in Pillow format."""
        return pyautogui.screenshot(region=self.loc)

    def send_key(self, key):
        pass


class AutomatableGame__Windows(AutomatableGame):
    def __init__(self):
        super().__init__(self)
        self.d = d3dshot.create()

    def fetch_sshot(self):
        return self.d.screenshot()  # TODO: Cut this to self.loc(x, y, w, h)
