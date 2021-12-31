from math import sqrt, cos, sin

def squared_distance(vec1, vec2):
    """returns distance-squared between two x, y point tuples"""
    return (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2

def rect_radius_squared(w, h):
    """Returns the radius^2 of the circle inscribed in a rectangle of w * h"""
    return (w/2)**2 + (h/2)**2

def point_in_rect(pt, rect):
    """Returns True if the (x,y) point is within the ((x,y),(w,h)) rectangle."""
    px, py = pt
    tl, wh = rect
    rx, ry = tl
    rw, rh = wh
    rx2 = rx + rw
    ry2 = ry + rh
    return all([px >= rx, py >= ry, px <= rx2, py <= ry2])

def polar_to_cartesian(theta, radius, center=(0, 0)):
    x = radius * cos(theta)
    y = radius * sin(theta)
    return (x + center[0], y + center[1])
