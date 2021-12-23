import gameio
import cv2
import numpy as np

from utility import *
import pointcluster

class GameModel:
    """Platform-independent representation of the game's state."""
    def __init__(self, io:gameio.AbstractGameIO):
        self.gameio = io
        self.asteroids = [
            ("big", cv2.imread("images/game_assets/rock-big.png", 0)),
            ("normal", cv2.imread("images/game_assets/rock-normal.png", 0)),
            ("small", cv2.imread("images/game_assets/rock-small.png", 0))
            ]
        self.ships = [
            ("ship_off", cv2.imread("images/game_assets/spaceship-off.png", 0)),
            ("ship_on", cv2.imread("images/game_assets/spaceship-on.png", 0))
            ]
        #self.missile = ("missile", cv2.imread("images/game_assets/missile.png", 0))
        self.frame = None
        self.cv_template_thresh = 0.6 # reconfigurable at runtime
        self.duplicate_dist_thresh = 36

    def with_frame(fn):
        """Decorator to process screenshot to cv2 format once upon first requirement, then reuse."""
        def inner(self, *args, **kwargs):
            if self.frame is None:
                #print("Fetching frame.")
                sshot = self.gameio.fetch_sshot()
                open_cv_image = np.array(sshot) 
                # Convert RGB to BGR 
                self.frame = open_cv_image[:, :, ::-1].copy()
                self.color_frame = np.copy(self.frame)
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
    def display_results(self, rects = [], pointsets = [], circles = []):
        """Draws results on the current frame for test purposes."""
        displayable = np.copy(self.color_frame)
        for pt, wh, label in rects:
            color = { "big":    (255, 0, 0),
                      "normal": (0, 255, 0),
                      "small":  (0, 0, 255),
                      "missile": (128, 0, 0),
                      "ship_on": (0, 0, 128),
                      "ship_off": (0, 64, 128)}[label]
            cv2.rectangle(displayable, pt, wh, color, 1)
            cv2.putText(displayable, label, pt,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0, color)
        for ps in pointsets:
            color = (0, 255, 255)
            cv2.polylines(displayable, np.int32([ps]), True, color)

        for center, radius, label in circles:
            color = (255, 255, 0)
            cv2.circle(displayable, np.int32(center), int(radius), color, 1)
            cv2.putText(displayable, label, np.int32(center),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, color)
        
        cv2.imshow("Results", displayable)
        cv2.waitKey(0)

    @with_frame
    def frame_sift(self):
        sift = cv2.SIFT_create()
        kp_desc = {} # dict of (keypoints, descriptions) for all ship sprites
        kp_desc["frame"] = sift.detectAndCompute(self.frame, None)
        frame_kp, frame_desc = kp_desc["frame"]
##        for label, s in self.ships:
##            kp_desc[label]  = sift.detectAndCompute(s, None)
##        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
##        matchsets = {}
##        for label in kp_desc:
##            _, desc = kp_desc[label]
##            matchsets[label] = bf.match(frame_desc, desc)
##        #return { "matchsets": matchsets,
##        #         "kp_desc": kp_desc
##        #       }
        ship_rsq = rect_radius_squared(*self.ships[0][1].shape)
        #print(f"max radius^2: {ship_rsq}")
        clusters = pointcluster.cluster_set([k.pt for k in frame_kp], sqrt(ship_rsq))

        return clusters

    @with_frame
    def find_ships(self):
        ship_rects = []
        for label, a in self.ships:
            h, w = a.shape
            res = cv2.matchTemplate(self.frame, a, cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= self.cv_template_thresh)
            for pt in zip(*loc[::-1]):
                if not ship_rects or squared_distance(ship_rects[-1][0], pt) > self.duplicate_dist_thresh:
                    ship_rects.append((pt, (pt[0] + w, pt[1] + h), label))
        return ship_rects

##    @with_frame
##    def find_missiles(self):
##        """This technique does not work for the 9x9 pixel missile image."""
##        missile_rects = []
##        label, img = self.missile
##        h, w = img.shape
##        res = cv2.matchTemplate(self.frame, img, cv2.TM_CCOEFF_NORMED)
##        loc = np.where( res >= self.cv_template_thresh)
##        for pt in zip(*loc[::-1]):
##            if not missile_rects or squared_distance(missile_rects[-1][0], pt) > self.duplicate_dist_thresh:
##                missile_rects.append((pt, (pt[0] + w, pt[1] + h), label))
##        return missile_rects

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
    a_results = gm.find_asteroids()
    print(f"Found {len(a_results)} asteroids")
    #for a in a_results:
    #    print(a[0]) # position tuple
    #gm.display_results(results)
    s_results = gm.frame_sift()
    ship_results = gm.find_ships()
    polygons = [c.points for c in s_results]
    #circles = [(c.center, c.max_distance, f"cluster_{i}") for i, c in enumerate(s_results)]
    r_circles = [(c.center, sqrt(rect_radius_squared(*gm.ships[0][1].shape)), f"cluster_{i}") for i, c in enumerate(s_results)]
    gm.display_results(rects=a_results+ship_results, pointsets=polygons, circles=r_circles)
