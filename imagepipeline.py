import cv2
import numpy as np

class CVImage:
    def __init__(self, label="", img=None, color=False, **kwargs):
        self.label = label
        self.image = img
        self.iscolor = color
        if kwargs:
            kwargs["color"] = color
            self.load(**kwargs)

    def load(self, filename, color=False, label=None):
        self.image = cv2.imread(filename, int(color))
        if label: self.label = label
        return self

    def copy(self):
        return np.copy(self.image)

    def snip(self, rect):
        assert all((len(rect)==2, len(rect[0])==2, len(rect[1])==2)) #((x,y),(w,h))
        return self.image[rect[0][0]:rect[1][0],
                          rect[0][1]:rect[1][1]]

    def mask(self, rect, mask_color=None, nonmask_color=None):
        assert all((len(rect)==2, len(rect[0])==2, len(rect[1])==2)) #((x,y),(w,h))
        if mask_color is None:
            mask_color = (0, 0, 0) if self.iscolor else 0
        if nonmask_color is None:
            nonnmask_color = (255, 255, 255) if self.iscolor else 255
        mask = np.full(self.image.shape, nonmask_color, dtype=np.uint8)
        cv2.rectangle(mask, *rect, color=mask_color, thickness=cv2.FILLED)
        return cv2.bitwise_and(self.image, mask)

    def sift_clusters(self, cluster_radius):
        sift = cv2.SIFT_create()
        keypoints, descriptions = sift.detectAndCompute(self.image, None)
        return pointcluster.cluster_set([k.pt for k in keypoints], cluster_radius)

    def blob_params(self, minThreshold = 10, maxThreshold = 200,
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
            p.filterByInertiaRatio = True
            if minInertiaRatio: p.minInertiaRatio = minInertiaRatio
            if maxInertiaRatio: p.maxInertiaRatio = maxInertiaRatio
        if minCircularity or maxCircularity:
            p.filterByCircularity = True
            if minCircularity: p.minCircularity = minCircularity
            if maxCircularity: p.maxCircularity = maxCircularity
        return p

    def blob_detect(self, params=None, invert=False):
        if params is None: params = self.blob_params()
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(cv2.bitwise_not(self.image) if invert else self.image)

    def show(self, delay=0):
        cv2.imshow(self.label, self.image)
        cv2.waitKey(delay)
    

class ImagePipeline:
    def __init__(self):
        pass

    
