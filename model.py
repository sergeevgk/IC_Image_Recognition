
element_types = ['computing element', 'radiator', 'connector', 'mount', 'electricity supply', 'capacitor']


class ElementCoordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        pass


class SquareElemCoord(ElementCoordinates):
    def __init__(self, x, y, width):
        super().__init__(x, y)
        self.w = width


class CircleElemCoord(ElementCoordinates):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.r = radius


class RectElemCoord(ElementCoordinates):
    def __init__(self, x, y, length, width):
        super().__init__(x, y)
        self.l = length
        self.w = width


class model:
    def __init__(self, ):
        self.d = d