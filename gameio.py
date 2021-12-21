#!/usr/bin/env python3
"""Module to interact with the game."""

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
        pass

    # feel free to change these two functions in the descendants

    def key_up(self, key: str):
        """Holds a key on a virtual keyboard."""
        pyautogui.keyUp(key)

    def key_down(self, key: str):
        """Lets go of a key on a virtual keyboard."""
        pyautogui.keyDown(key)

    def send_key(self, key: str):
        """Presses a key on a virtual keyboard."""
        pyautogui.press(key)


class WindowsGameIO(AbstractGameIO):
    def __init__(self):
        super().__init__()
        self.d = d3dshot.create()

    def fetch_sshot(self):
        return self.d.screenshot()  # TODO: Cut this to self.loc(x, y, w, h)


class LinuxGameIO(AbstractGameIO):
    def __init__(self):
        super().__init__()

    def fetch_sshot(self):
        return pyautogui.screenshot(region=self.loc)
