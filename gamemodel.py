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
            CVImage("big", color = False, filename = "images/game_assets/rock-big.png"),
            CVImage("normal", color = False, filename = "images/game_assets/rock-normal.png"),
            CVImage("small", color = False, filename = "images/game_assets/rock-small.png")
            ]
        self.ships = [
            ("ship_off", cv2.imread("images/game_assets/spaceship-off.png", 0)),
            ("ship_on", cv2.imread("images/game_assets/spaceship-on.png", 0))
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
##                #print("Fetching frame.")
##                sshot = self.gameio.fetch_sshot()
##                open_cv_image = np.array(sshot) 
##                # Convert RGB to BGR 
##                self.frame = open_cv_image[:, :, ::-1].copy()
##                self.color_frame = np.copy(self.frame)
##                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
##                self.mask_frame()
                self.color_frame = CVImage("gameio frame")
                self.color_frame.from_pil(self.gameio.fetch_sshot())
                self.frame = CVImage("BW frame", self.color_frame.copy())
                self.frame.image = self.frame.convert_color(False)
                print(self.frame)
                self.mask_frame()
            return fn(self, *args, **kwargs)
        return inner

    def mask_frame(self):
##        self.lives_img = self.frame[self.lives_rect[0][0]:self.lives_rect[0][1],
##                                    self.lives_rect[1][0]:self.lives_rect[1][1]]
##        lives_mask = np.full(self.frame.shape, 255, dtype=np.uint8)
##        
##        cv2.rectangle(lives_mask,
##                      *self.lives_rect,
##                      color=0, thickness=cv2.FILLED)
##
##        self.score_img = self.frame[self.score_rect[0][0]:self.score_rect[0][1],
##                                    self.score_rect[1][0]:self.score_rect[1][1]]
##        score_mask = np.full(self.frame.shape, 255, dtype=np.uint8)
##        cv2.rectangle(score_mask,
##                      *self.score_rect,
##                      color = 0, thickness=cv2.FILLED)
##        self.frame = cv2.bitwise_and(self.frame, lives_mask)
##        self.frame = cv2.bitwise_and(self.frame, score_mask)

        self.lives_img = CVImage("lives", self.frame.snip(self.lives_rect))
        self.frame.image = self.frame.mask(self.lives_rect)
        self.score_img = CVImage("score", self.frame.snip(self.score_mask))
        self.frame.image = self.frame.mask(self.score_mask)
                
    def clear_frame(self):
        self.prev_frame = frame
        self.frame = None

    @with_frame
    def find_asteroids(self):
##        asteroid_rects = []
##        for label, a in self.asteroids:
##            h, w = a.shape
##            res = cv2.matchTemplate(self.frame, a, cv2.TM_CCOEFF_NORMED)
##            loc = np.where( res >= self.cv_template_thresh)
##            for pt in zip(*loc[::-1]):
##                if not asteroid_rects or squared_distance(asteroid_rects[-1][0], pt) > self.duplicate_dist_thresh:
##                    asteroid_rects.append((pt, (pt[0] + w, pt[1] + h), label))
##        return asteroid_rects
        results = [self.frame.template_detect(i,
                   self.cv_template_thresh,
                   self.duplicate_dist_thresh)
                   for i in self.asteroids]

    @with_frame
    def display_results(self, rects = [], pointsets = [], circles = []):
        """Draws results on the current frame for test purposes."""
##        displayable = np.copy(self.color_frame)
##        cv2.rectangle(displayable, *self.lives_rect, (255,255,255), 1)
##        cv2.rectangle(displayable, *self.score_rect, (255,255,255), 1)
        displayable = CVImage("GameModel results", self.color_frame.copy())
        #else:
        #    displayable = np.copy(self.color_frame)
        label_color = { "big":    (255, 0, 0),
                      "normal": (0, 255, 0),
                      "small":  (0, 0, 255),
                      "missile": (0, 255, 128),
                      "ship_on": (0, 0, 128),
                      "ship_off": (0, 64, 128)}
        for r in rects:
##            cv2.rectangle(displayable, pt, wh, color, 1)
##            cv2.putText(displayable, label, pt,
##                    cv2.FONT_HERSHEY_PLAIN,
##                    1.0, color)
            displayable.draw_rect(r, color=label_color[r.label])
        for ps in pointsets:
##            color = (0, 255, 255)
##            cv2.polylines(displayable, np.int32([ps]), True, color)
            displayable.draw_poly(ps, color=(0, 255, 255))

        for center, radius, label in circles:
##            color = (255, 255, 0)
##            cv2.circle(displayable, np.int32(center), int(radius), color, 1)
##            cv2.putText(displayable, label, np.int32(center),
##                        cv2.FONT_HERSHEY_PLAIN,
##                        1.0, color)
            displayable.draw_circle(center, radius)
            displayable.draw_text(label, center, (255, 255, 0))
        
        cv2.imshow("Results", displayable)
        cv2.waitKey(0)

    @with_frame
    def frame_sift(self):
##        sift = cv2.SIFT_create()
##        kp_desc = {} # dict of (keypoints, descriptions) for all ship sprites
##        kp_desc["frame"] = sift.detectAndCompute(self.frame, None)
##        frame_kp, frame_desc = kp_desc["frame"]
        
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
        ship_r = sqrt(rect_radius_squared(*self.ships[0][1].shape) * 0.85)
        #print(f"max radius^2: {ship_rsq}")
        #clusters = pointcluster.cluster_set([k.pt for k in frame_kp], sqrt(ship_rsq))
        #return clusters
        return self.frame.sift_clusters(cluster_radius = ship_r)

    @with_frame
    def find_ships(self):
##        ship_rects = []
##        for label, a in self.ships:
##            h, w = a.shape
##            res = cv2.matchTemplate(self.frame, a, cv2.TM_CCOEFF_NORMED)
##            loc = np.where( res >= self.cv_template_thresh)
##            for pt in zip(*loc[::-1]):
##                if not ship_rects or squared_distance(ship_rects[-1][0], pt) > self.duplicate_dist_thresh:
##                    ship_rects.append((pt, (pt[0] + w, pt[1] + h), label))
##        return ship_rects
        return [self.frame.template_detect(a, self.cv_template_thresh, self.duplicate_dist_thresh) for a in self.ships]

    @with_frame
    def find_missiles(self):
##        # Setup SimpleBlobDetector parameters.
##        params = cv2.SimpleBlobDetector_Params()
##
##        # Change thresholds
##        params.minThreshold = 10;
##        params.maxThreshold = 200;
##
##        # Filter by Area.
##        params.filterByArea = True
##        #params.minArea = 1500
##        params.maxArea = 100
##
##        # Filter by Circularity
##        #params.filterByCircularity = True
##        #params.minCircularity = 0.1
##
##        # Filter by Convexity
##        params.filterByConvexity = True
##        params.minConvexity = 0.95
##
##        # Filter by Inertia
##        params.filterByInertia = True
##        params.minInertiaRatio = 0.4
##
##        detector = cv2.SimpleBlobDetector_create(params)
##        keypoints = detector.detect(cv2.bitwise_not(self.frame)) # inverted black/white frame

        p = CVImage.blob_params(minThreshold = 10, maxThreshold = 200,
                                maxArea = 100,
                                minConvexity = 0.95,
                                minIntertiaRatio = 0.4)
        return self.frame.blob_detect(size=9, params=p)
        #im_with_keypoints = cv2.drawKeypoints(self.frame, keypoints, np.array([]),
        #                    (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("keypoints", im_with_keypoints)
        #cv2.waitKey(0)
##        s = 9 # pixels for the missile
##        rect_tuple = lambda pt: ((int(pt[0]-s/2),int(pt[1]-s/2)),
##                                 (int(pt[0]+s/2), int(pt[1]+s/2)),
##                                 "missile")
##        return [rect_tuple(k.pt) for k in keypoints]

    def analyse_frame(self):
        rocks = self.find_asteroids()
        #lives = self.find_ships()
        shots = self.find_missiles()
        clusters = self.frame_sift()

        labeled_objects = rocks + shots
        mystery_clusters = []
        # TODO: remove these comprehensions and document pretty utility functions.
        easy_find = lambda cluster: any(
            [(not cluster.max_distance or cluster.max_distance < max(lo[1][0] - lo[0][0], lo[1][1] - lo[0][1]))
             and point_in_rect(cluster.center, (lo[0], lo[1]))
             for lo in labeled_objects])
        hard_find = lambda cluster: any(
            [(not cluster.max_distance or cluster.max_distance < max(lo[1][0] - lo[0][0], lo[1][1] - lo[0][1]))
            and all([point_in_rect(p, (lo[0], lo[1]))
            for p in cluster.points])
            for lo in labeled_objects])
        # Allow me to explain/apologize.
        ## The first term (cluster.max_distance < ...) stops big point clusters from
        ## being regarded as smalll objects. (Player ship being matched "inside" a missile)

        ## The second term (point_in_rect(...)) checks for a "cluster" inside a "rect".
        ## easy_find just checks the center.
        ## hard_find checks every point, in case the center is off.

        for i, c in enumerate(clusters):
            #if easy_find(c): continue
            if hard_find(c): continue
            mystery_clusters.append(c)
        r_circles = [(c.center, c.max_distance or 5, f"mystery_{i}") for i, c in enumerate(mystery_clusters)]
        gm.display_results(rects=labeled_objects, circles=r_circles)

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
    r_circles = [(c.center, sqrt(rect_radius_squared(*gm.ships[0][1].shape)), f"cluster_{i}") for i, c in enumerate(s_results)]
    missile_results = gm.find_missiles()
    ##m_circles = [(pt, 10, f"missile_{i}") for i, pt in enumerate(missiles)]
    ##pprint(a_results+ship_results+missile_results)
    gm.display_results(rects=a_results+ship_results+missile_results, pointsets=polygons, circles=r_circles)
    gm.analyse_frame()
