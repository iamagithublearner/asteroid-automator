class Rect:
    def __init__(self, *args, label=None, **kwargs):
        if len(args) == 4 and all([type(i) is int or type(i) is float for i in args]):
            self.x, self.y, self.w, self.h = args
        elif len(args) == 2 and all([type(i) is tuple and len(i) == 2 and all([type(j) is int or type(j) is float for j in i]) for i in args]):
            xy, wh = self.args
            self.x, self.y = xy
            self.w, self.h = wh
        elif all([k in kwargs for k in ("x", "y", "w", "h")]):
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.w = kwargs["w"]
            self.h = kwargs["h"]
        elif all([k in kwargs for k in ("x", "y", "x2", "y2")]):
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.w = kwargs["x2"] - self.x
            self.h = kwargs["y2"] - self.y
        elif all([k in kwargs for k in ("x1", "y1", "x2", "y2")]):
            self.x = kwargs["x1"]
            self.y = kwargs["y1"]
            self.w = kwargs["x2"] - self.x
            self.h = kwargs["y2"] - self.y
        else:
            raise RuntimeError("Rect requires 4 values: two coordinates or a coordinate plus width and height.")
        self.label = label
        
    def __repr__(self):
        return f"<Rect label={repr(self.label)}, (({self.x}, {self.y}), ({self.w}, {self.h}))>"

    def __iter__(self):
        yield (self.x, self.y)
        yield (self.w, self.h)

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0: return (self.x, self.y)
        elif i == 1: return (self.w, self.h)
        else: raise IndexError("Rect only supports index of 0 or 1.")

    def __setitem__(self, i, value):
        assert i in (0, 1) and len(value) == 2
        if not i: self.x, self.y = value
        else: self.w, self.h = value

    @property
    def point(self):
        return (self.x, self.y)

    @property
    def point2(self):
        return (self.x + self.w, self.y + self.h)
