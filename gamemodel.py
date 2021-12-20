import gameio
import cv2
import numpy as np

class GameModel:
    """Platform-independent representation of the game's state."""
    def __init__(self, io:gameio.AbstractGameIO):
        self.gameio = io
        self.asteroids = [cv2.imread("images/game_assets/rock-big.png", 0),
                          cv2.imread("images/game_assets/rock-normal.png", 0),
                          cv2.imread("images/game_assets/rock-small.png", 0)
                         ]
        self.frame = None
        self.thresh = 0.6 # reconfigurable at runtime

    def with_frame(fn):
        """Decorator to process screenshot to cv2 format once upon first requirement, then reuse."""
        def inner(self):
            if self.frame is None:
                print("Fetching frame.")
                sshot = self.gameio.fetch_sshot()
                open_cv_image = np.array(sshot) 
                # Convert RGB to BGR 
                self.frame = open_cv_image[:, :, ::-1].copy()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            return fn(self)
        return inner

    def clear_frame(self):
        self.frame = None

    @with_frame
    def find_asteroids(self):
        asteroid_rects = []
        displayable = np.copy(self.frame)
        for a in self.asteroids:
            h, w = a.shape
            res = cv2.matchTemplate(self.frame, a, cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= self.thresh)
            ## Example code for displaying detected asteroid locations
            #for pt in zip(*loc[::-1]):
            #    cv2.rectangle(displayable, pt, (pt[0] + w, pt[1] + h), 255, 1)
        #cv2.imshow("Found asteroids", displayable)
        #cv2.waitKey(0)

if __name__ == '__main__':
    import platform
    if platform.system() == "Windows":
        io = gameio.WindowsGameIO()
    #TODO, detect OSX or show a message of sadness
    else:
        io = gameio.LinuxGameIO()
    input("Press <enter> to locate the game at the start screen.")
    gm = GameModel(io)
    input("Press <enter> to detect asteroids on screen.")
    gm.find_asteroids()
