from utility import *
from shapes import Rect

class PointCluster:
    def __init__(self):
        self.points = []
        self.center = (0, 0)
        self.max_distance = None

    def update(self):
        if len(self.points) == 0: return
        self.center = (sum([p[0] for p in self.points]) / len(self.points),
                       sum([p[1] for p in self.points]) / len(self.points))
        self.max_distance = sqrt(max(
            [squared_distance(self.center, p) for p in self.points]))

    def add(self, pt):
        self.points.append(pt)
        self.update()

    def pop(self):
        p = self.points.pop(-1)
        self.update()
        return p

    def __repr__(self):
        c = f"({self.center[0]:.1f},{self.center[1]:.1f})"
        return f"<PointCluster center={c}, {len(self.points)} points>"

    def bounding_rect(self):
        """Returns the smallest rectangle that contains all the cluster's points."""
        top = min(p[1] for p in self.points)
        height = max(p[1] for p in self.points) - top
        left = min(p[0] for p in self.points)
        width = max(p[0] for p in self.points) - left
        return Rect((left, top), (width, height))

def cluster_set(points, maxradius):
    """returns a list of PointCluster objects. Points are fit within circles of maxradius"""
    clusters = []
    for pt in points:
        if len(clusters) == 0:
            #print("first cluster")
            clusters.append(PointCluster())
            clusters[-1].add(pt)
            continue
        # add point to its nearest cluster
        scored_clusters = [(c, squared_distance(pt, c.center)) for c in clusters]
        scored_clusters.sort(key=lambda i: i[1])
        winner = scored_clusters[0][0]
        winner.add(pt)
        
        # if maxradius constraint was violated, pop the newest point & add new cluster
        if winner.max_distance > maxradius:
            #print(f"{winner.max_distance} > {maxradius}; new cluster")
            winner.pop()
            clusters.append(PointCluster())
            clusters[-1].add(pt)
            
    # refine step - accept centers as fixed, put points in closest center
    new_clusters = {c.center: PointCluster() for c in clusters}
    closest = lambda pt: sorted(new_clusters.keys(), key= lambda i: squared_distance(pt, i))[0]
    for point in points:
        new_clusters[closest(point)].add(point)
    #print(clusters)
    #print(new_clusters.values())
    return new_clusters.values()

def cluster_overlaps_rect(cluster, rect):
    if not cluster.max_distance:
        # cluster is a single point
        return point_in_rect(cluster.center, rect)
    elif not point_in_rect(cluster.center, rect):
        # center is outside the rect, so >= half the points are outside of it.
        return False
    largest_dimension = max(rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
    if cluster.max_distance < largest_dimension:
        return True

def cluster_within_rect(cluster, rect):
    if not cluster.max_distance:
        return point_in_rect(cluster.center, rect)
    return all([point_in_rect(p, rect) for p in cluster.points])
