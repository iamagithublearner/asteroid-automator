import cv2
import numpy as np
import typing
import pointcluster

from shapes import Rect
from utility import *

class CVImage:
    """Dummy definition to allow recursive type hints"""
    pass

class CVImage:
    def __init__(self, label="", img:np.ndarray=None, **kwargs):
        """You can provide a 'filename' keyword arg to automatically load a file."""
        self.label = label
        self.image = img
        self._init_kwargs = kwargs
        if kwargs:
            load_kwargs = dict(kwargs) # copy
            self.load(**load_kwargs)

    @property
    def is_color(self):
        return len(self.image.shape) == 3

    def load(self, filename:str, label:str=None):
        """Load an image from file. You can optionally set the 'label' keyword."""
        self.image = cv2.imread(filename)
        if label: self.label = label
        return self

    def convert_color(self, color:bool):
        return cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR if color else cv2.COLOR_BGR2GRAY)
        
    def __repr__(self):
        if self._init_kwargs:
            kwargstr = ", " + ", ".join([f"{k}={repr(self._init_kwargs[k])}" for k in self._init_kwargs])
        else:
            kwargstr = ''
        return f"<CVImage label={repr(self.label)}, image={self.image.shape} px, is_color={self.is_color}{kwargstr}>"

    def copy(self):
        return np.copy(self.image)

    def snip(self, rect):
        assert all((len(rect)==2, len(rect[0])==2, len(rect[1])==2)) #((x,y),(w,h))
        return self.image[rect[0][1]:rect[0][1]+rect[1][1],
                          rect[0][0]:rect[0][0]+rect[1][0]
                          ]

    def mask(self, rect, mask_color=None, nonmask_color=None):
        assert all((len(rect)==2, len(rect[0])==2, len(rect[1])==2)) #((x,y),(w,h))
        if mask_color is None:
            mask_color = (0, 0, 0) if self.is_color else 0
        if nonmask_color is None:
            nonmask_color = (255, 255, 255) if self.is_color else 255
        mask = np.full(self.image.shape, nonmask_color, dtype=np.uint8)
        cv2.rectangle(mask, *rect, color=mask_color, thickness=cv2.FILLED)
        return cv2.bitwise_and(self.image, mask)

    def sift_clusters(self, cluster_radius) -> pointcluster.PointCluster:
        sift = cv2.SIFT_create()
        keypoints, descriptions = sift.detectAndCompute(self.image, None)
        return pointcluster.cluster_set([k.pt for k in keypoints], cluster_radius)

    @staticmethod
    def blob_params(*, minThreshold = 10, maxThreshold = 200,
                     minArea = None, maxArea = None,
                     minCircularity = None, maxCircularity = None,
                     minConvexity = None, maxConvexity = None,
                     minInertiaRatio = None, maxInertiaRatio = None):
        p = cv2.SimpleBlobDetector_Params()
        p.minThreshold = minThreshold
        p.maxThreshold = maxThreshold
        if minArea or maxArea:
            p.filterByArea = True
            if minArea: p.minArea = minArea
            if maxArea: p.maxArea = maxArea
        if minConvexity or maxConvexity:
            p.filterByConvexity = True
            if minConvexity: p.minConvexity = minConvexity
            if maxConvexity: p.maxConvexity = maxConvexity
        if minInertiaRatio or maxInertiaRatio:
            p.filterByInertia = True
            if minInertiaRatio: p.minInertiaRatio = minInertiaRatio
            if maxInertiaRatio: p.maxInertiaRatio = maxInertiaRatio
        if minCircularity or maxCircularity:
            p.filterByCircularity = True
            if minCircularity: p.minCircularity = minCircularity
            if maxCircularity: p.maxCircularity = maxCircularity
        return p

    def blob_detect(self, size:int, params=None, invert:bool=False, label:str=None) -> typing.List[Rect]:
        if params is None: params = CVImage.blob_params()
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cv2.bitwise_not(self.image) if invert else self.image)
        rects = []
        s = size / 2.0
        for kp in keypoints:
            rects.append(Rect(x=kp.pt[0] - s, y = kp.pt[1] - s,
                              w = size, h = size,
                              label = label or "blob"))
        return rects

    def template_detect(self, template:CVImage, threshold:int, dupe_spacing:int) -> typing.List[Rect]:
        if template.is_color:
            h, w, _ = template.image.shape
        else:
            h, w = template.image.shape
        if template.is_color != self.is_color:
            template = CVImage(template.label, template.convert_color(not template.is_color))
        res = cv2.matchTemplate(self.image, template.image, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        rects = []
        for pt in zip(*loc[::-1]):
            if len(rects) > 0:
                if squared_distance(rects[-1][0], pt) < dupe_spacing: continue
            rects.append(Rect(x=pt[0], y=pt[1], w=w, h=h, label=template.label))
        return rects

    def show(self, delay=0):
        cv2.imshow(self.label, self.image)
        cv2.waitKey(delay)

    def draw_rect(self, rect:Rect, color=None, text_color=None, text:bool=True, thickness=1):
        if color is None:
            color = (255, 255, 255) if self.is_color else 255
        cv2.rectangle(self.image, rect.point, rect.point2, color, thickness)
        if text:
            self.draw_text(rect.label, rect.point, text_color if text_color else color)

    def draw_poly(self, points:typing.List[typing.Tuple], closed=True, color=None):
        if color is None:
            color = (255, 255, 255) if self.is_color else 255
        cv2.polylines(self.image, np.int32([points]), closed, color)

    def draw_circle(self, center, radius, color=None, thickness = 1):
        if color is None:
            color = (255, 255, 255) if self.is_color else 255
        cv2.circle(self.image, np.int32(center), np.int32(radius), color, thickness)

    def draw_text(self, text, point, color):
        cv2.putText(self.image, text, np.int32(point), cv2.FONT_HERSHEY_PLAIN, 1.0, color)
    

class ImagePipeline:
    def __init__(self):
        pass

# running this module executes tests
if __name__ == '__main__':
    # initializer for CVImage can load from file
    img = CVImage("test frame", filename="/home/john/Desktop/Screenshot at 2021-12-19 20-55-22.png")
    #img.show()
    # initializer for CVImage can accept a numpy array
    img_no_title = CVImage("test frame", img.snip( ((0,24),(800,600)) ))
    #img_no_title.show()

    #standard rectangle format used throughout the class, avoiding ugly splat operator
    lives_rect = ((10,10), (190, 65))
    lives = CVImage("lives", img_no_title.snip(lives_rect))
    lives.show()

    
