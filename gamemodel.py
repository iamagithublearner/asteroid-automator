import gameio
import cv2
import numpy as np
from functools import wraps

from utility import *
import pointcluster
from imagepipeline import CVImage, ImagePipeline

class GameModel:
    """Platform-independent representation of the game's state."""
    def __init__(self, io:gameio.AbstractGameIO):
        self.gameio = io
        self.asteroids = [
            CVImage("big", filename = "images/game_assets/rock-big.png"),
            CVImage("normal", filename = "images/game_assets/rock-normal.png"),
            CVImage("small", filename = "images/game_assets/rock-small.png")
            ]
        self.ships = [
            CVImage("ship_off", filename = "images/game_assets/spaceship-off.png"),
            CVImage("ship_on", filename = "images/game_assets/spaceship-on.png")
            ]
        #self.missile = ("missile", cv2.imread("images/game_assets/missile.png", 0))
        self.frame = None
        self.prev_frame = None
        self.color_frame = None
        self.score_img = None
        self.lives_img = None
        self.lives_rect = ((10,10), (190, 65))
        self.score_rect = ((600, 25), (780, 65))
        self.cv_template_thresh = 0.6 # reconfigurable at runtime
        self.duplicate_dist_thresh = 36

    def with_frame(fn):
        """Decorator to process screenshot to cv2 format once upon first requirement, then reuse."""
        @wraps(fn)
        def inner(self, *args, **kwargs):
            if self.frame is None:
                sshot = self.gameio.fetch_sshot()
                open_cv_image = np.array(sshot) 
                # Convert RGB to BGR 
                array = open_cv_image[:, :, ::-1].copy()
                self.color_frame = CVImage("gameio frame", np.copy(array))
                self.frame = CVImage("BW frame", self.color_frame.copy())
                self.frame.image = self.frame.convert_color(False)
                self.mask_frame()
            return fn(self, *args, **kwargs)
        return inner

    def mask_frame(self):
        self.lives_img = CVImage("lives", self.frame.snip(self.lives_rect))
        self.frame.image = self.frame.mask(self.lives_rect)
        self.score_img = CVImage("score", self.frame.snip(self.score_rect))
        self.frame.image = self.frame.mask(self.score_rect)
                
    def clear_frame(self):
        self.prev_frame = frame
        self.frame = None

    @with_frame
    def find_asteroids(self):
        results = []
        for a in self.asteroids:
            r = self.frame.template_detect(a,
               self.cv_template_thresh,
               self.duplicate_dist_thresh)
            results.extend(r)
        return results

    @with_frame
    def display_results(self, rects = [], pointsets = [], circles = [], label="GameModel Resuls"):
        """Draws results on the current frame for test purposes."""
        displayable = CVImage(label, self.color_frame.copy())
        label_color = { "big":    (255, 0, 0),
                      "normal": (0, 255, 0),
                      "small":  (0, 0, 255),
                      "missile": (0, 255, 128),
                      "ship_on": (0, 0, 128),
                      "ship_off": (0, 64, 128)}
        for r in rects:
            displayable.draw_rect(r, color=label_color.get(r.label, (128, 128, 128)))
        for ps in pointsets:
            displayable.draw_poly(ps, color=(0, 255, 255))

        for center, radius, label in circles:
            displayable.draw_circle(center, radius)
            displayable.draw_text(label, center, (255, 255, 0))

        displayable.show()

    @with_frame
    def frame_sift(self):
        ship_r = sqrt(rect_radius_squared(*self.ships[0].image.shape[:2]) * 0.85)
        return self.frame.sift_clusters(cluster_radius = ship_r)

    @with_frame
    def find_ships(self):
        results = []
        for a in self.ships:
            r = self.frame.template_detect(a,
                    self.cv_template_thresh,
                    self.duplicate_dist_thresh)
            results.extend(r)
        return results

    @with_frame
    def find_missiles(self, size=9):
        p = CVImage.blob_params(minThreshold = 10, maxThreshold = 200,
                                maxArea = 100,
                                minConvexity = 0.95,
                                minInertiaRatio = 0.4)
        return self.frame.blob_detect(size=size, params=p, invert=True)

    def analyse_frame(self):
        rocks = self.find_asteroids()
        #lives = self.find_ships()
        shots = self.find_missiles()
        clusters = self.frame_sift()

        labeled_objects = rocks + shots
        mystery_clusters = []
        easy_find = lambda cl: any([pointcluster.cluster_overlaps_rect(cl, lo)
            for lo in labeled_objects])
        hard_find = lambda cl: any([pointcluster.cluster_within_rect(cl, lo)
            for lo in labeled_objects])

        for i, c in enumerate(clusters):
            #if easy_find(c): continue
            if hard_find(c): continue
            mystery_clusters.append(c)
        #r_circles = [(c.center, c.max_distance or 5, f"mystery_{i}") for i, c in enumerate(mystery_clusters)]
        #gm.display_results(rects=labeled_objects, circles=r_circles)
        for i, c in enumerate(mystery_clusters):
            r = c.bounding_rect()
            r.label = f"mystery_{i}"
            labeled_objects.append(r)
        gm.display_results(rects=labeled_objects)

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

    from pprint import pprint

    #input("Press <enter> to detect asteroids on screen.")
    a_results = gm.find_asteroids()
    print(f"Found {len(a_results)} asteroids")
    #for a in a_results:
    #    print(a[0]) # position tuple
    #gm.display_results(results)
    s_results = gm.frame_sift()
    ship_results = gm.find_ships()
    polygons = [c.points for c in s_results]
    ##circles = [(c.center, c.max_distance, f"cluster_{i}") for i, c in enumerate(s_results)]
    r_circles = [(c.center, sqrt(rect_radius_squared(*gm.ships[0].image.shape[:2])), f"cluster_{i}") for i, c in enumerate(s_results)]
    missile_results = gm.find_missiles()
    ##m_circles = [(pt, 10, f"missile_{i}") for i, pt in enumerate(missiles)]
    ##pprint(a_results+ship_results+missile_results)
    rects = a_results
    if ship_results: rects.extend(ship_results)
    if missile_results: rects.extend(missile_results)
    #gm.display_results(rects=rects, pointsets=polygons, circles=r_circles)
    gm.analyse_frame()
