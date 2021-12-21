import gameio
import cv2
import numpy as np

def squared_distance(vec1, vec2):
    """returns distance-squared between two x, y point tuples"""
    return (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2

class GameModel:
    """Platform-independent representation of the game's state."""
    def __init__(self, io:gameio.AbstractGameIO):
        self.gameio = io
        self.asteroids = [
            ("big", cv2.imread("images/game_assets/rock-big.png", 0)),
            ("normal", cv2.imread("images/game_assets/rock-normal.png", 0)),
            ("small", cv2.imread("images/game_assets/rock-small.png", 0))
            ]
        self.frame = None
        self.cv_template_thresh = 0.6 # reconfigurable at runtime
        self.duplicate_dist_thresh = 10

    def with_frame(fn):
        """Decorator to process screenshot to cv2 format once upon first requirement, then reuse."""
        def inner(self, *args, **kwargs):
            if self.frame is None:
                print("Fetching frame.")
                sshot = self.gameio.fetch_sshot()
                open_cv_image = np.array(sshot) 
                # Convert RGB to BGR 
                self.frame = open_cv_image[:, :, ::-1].copy()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            return fn(self, *args, **kwargs)
        return inner

    def clear_frame(self):
        self.frame = None

    @with_frame
    def find_asteroids(self):
        asteroid_rects = []
        for label, a in self.asteroids:
            h, w = a.shape
            res = cv2.matchTemplate(self.frame, a, cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= self.cv_template_thresh)
            for pt in zip(*loc[::-1]):
                if not asteroid_rects or squared_distance(asteroid_rects[-1][0], pt) > self.duplicate_dist_thresh:
                    asteroid_rects.append((pt, (pt[0] + w, pt[1] + h), label))
        return asteroid_rects

    @with_frame
    def display_results(self, results):
        """Draws results on the current frame for test purposes."""
        displayable = np.copy(self.frame)
        for pt, wh, label in results:
            cv2.rectangle(displayable, pt, wh, 255, 1)
            cv2.putText(displayable, label, pt,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0, 255)
        cv2.imshow("Results", displayable)
        cv2.waitKey(0)


if __name__ == '__main__':
    import platform

    if platform.system() == "Windows":
        io = gameio.WindowsGameIO()
    # TODO: Detect OSX or show a message of sadness

    else:
        io = gameio.LinuxGameIO()

    #input("Press <enter> to locate the game at the start screen.")
    gm = GameModel(io)

    # for testing purposes, populating window location at top-left of my screen
    # io.loc is None when the title screen isn't found.
    # manually setting io.loc crops all screenshots as if the title was found.
    import pyscreeze
    io.loc = pyscreeze.Box(0, 25, 800, 599)

    #input("Press <enter> to detect asteroids on screen.")
    results = gm.find_asteroids()
    print(f"Found {len(results)} asteroids")
    for a in results:
        print(a[0]) # position tuple
    gm.display_results(results)
