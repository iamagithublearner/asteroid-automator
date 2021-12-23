from math import sqrt

def squared_distance(vec1, vec2):
    """returns distance-squared between two x, y point tuples"""
    return (vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2

def rect_radius_squared(w, h):
    """Returns the radius^2 of the circle inscribed in a rectangle of w * h"""
    return (w/2)**2 + (h/2)**2
