import cv2
import numpy as np

class CVImage:
    def __init__(self, label="", img=None, iscolor=False):
        self.label = label
        self.image = img
        self.iscolor = iscolor

    def load(self, filename, color=False, label=None):
        self.image = cv2.imread(filename, int(color))
        if label: self.label = label

    def copy(self):
        return np.copy(self.image)

##    def snip(self, point, width_height):
##        return self.image[self.point[0]:self.point[0]+self.width_height[0],
##                          self.point[1]:self.point[1]+self.width_height[1]]

    def snip(self, rect):
        assert all((len(rect)==2, len(rect[0])==2, len(rect[1])==2)) #((x,y),(w,h))
        return self.image[self.rect[0][0]:self.rect[0][1],
                          self.rect[1][0]:self.rect[1][1]]

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

    def set_blob_params(self, minThreshold = 10, maxThreshold = 200,
                     minArea = None, maxArea = None,
                     minCircularity = None, maxCircularity = None,
                     minConvexity = None, maxConvexity = None,
                     minInertiaRatio = None, maxInertiaRatio = None):
        p = cv2.SimpleBlobDetector_Params()
        

    def blob_detect(self, params=None):
        pass
    

class ImagePipeline:
    def __init__(self):
        pass

    
