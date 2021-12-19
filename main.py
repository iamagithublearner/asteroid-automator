#!/usr/bin/env python3
"""Main asteroid-automator script."""

import platform
from abc import ABC, abstractmethod

import pyautogui

PLATFORM_SYS = platform.system()

if PLATFORM_SYS == "Windows":
    import d3dshot


class AbstractGameIO(ABC):
    """Base class for each platform."""

    @abstractmethod
    def __init__(self):
        """Initializes components needed for the I/O to operate."""
        self.loc = pyautogui.locateOnScreen("images/app.png")

    @abstractmethod
    def fetch_sshot(self):
        """Creates a screenshot, and returns it in Pillow format."""
        return pyautogui.screenshot(region=self.loc)

    @abstractmethod
    def send_key(self, key: str):
        """Presses a key on a virtual keyboard."""
        pass


class WindowsGameIO(AbstractGameIO):
    def __init__(self):
        super().__init__(self)
        self.d = d3dshot.create()

    def fetch_sshot(self):
        return self.d.screenshot()  # TODO: Cut this to self.loc(x, y, w, h)

    def send_key(self, key: str):
        pass  # TODO: Add send_key method


class LinuxGameIO(AbstractAutomatableGame):
    """Base class for each platform."""

    def __init__(self):
        super().__init__(self)

    def fetch_sshot(self):
        return pyautogui.screenshot(region=self.loc)

    def send_key(self, key: str):
        pass  # TODO: Add send_key method
