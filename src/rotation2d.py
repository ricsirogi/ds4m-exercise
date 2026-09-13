class Rotation2D:
    def __init__(self, point: tuple[float, float], angle: float):
        self.point = point
        self.angle = angle

    def __repr__(self):
        return "Rotation2D(point=..., angle=...)"
