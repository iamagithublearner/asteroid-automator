from utility import *

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
