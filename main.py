import pyautogui
import os

if os.name == 'windows':
    import d3dshot

class AutomatableGame:
    def __init__(self):
        self.loc = pyautogui.locateOnScreen("images/app.png")
        
    def fetch_sshot(self):
        return pyautogui.screenshot(region=self.loc)

    def send_key(self, key):
        pass


class AutomatableGame__Windows(AutomatableGame):
    def __init__(self):
        super().__init__(self)
        self.d = d3dshot.create()
        
    def fetch_sshot(self):
        return d.screenshot() # Cut this to self.loc(x, y, w, h)
