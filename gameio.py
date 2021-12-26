#!/usr/bin/env python3
"""Module to interact with the game."""

import platform
from abc import ABC, abstractmethod

import pyautogui

PLATFORM_SYS = platform.system()


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

    @abstractmethod
    def key_up(self, key: str):
        """Holds a key on a virtual keyboard."""
        pyautogui.keyUp(key)

    @abstractmethod
    def key_down(self, key: str):
        """Lets go of a key on a virtual keyboard."""
        pyautogui.keyDown(key)

    def send_key(self, key: str):
        """Presses a key on a virtual keyboard."""
        self.key_up(key)
        self.key_down(key)


class WindowsGameIO(AbstractGameIO):
    def __init__(self):
        super().__init__()

    def fetch_sshot(self):
        return pyautogui.screenshot(region=self.loc)

    def key_up(self, key: str):
        pyautogui.keyUp(key)

    def key_down(self, key: str):
        pyautogui.keyDown(key)


class LinuxGameIO(AbstractGameIO):
    def __init__(self):
        super().__init__()

    def fetch_sshot(self):
        return pyautogui.screenshot(region=self.loc)
