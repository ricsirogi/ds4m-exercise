from __future__ import annotations

import numpy as np
import pandas as pd


class Rotation2D:
    def __init__(self, point: tuple[float, float], angle: float):
        self.point = point
        self.angle = angle

        self.__transformation_matrix = None

    @property
    def transformation_matrix(self):
        first_row = [np.cos(self.angle), -np.sin(self.angle)]
        second_row = [np.sin(self.angle), np.cos(self.angle)]

        self.__transformation_matrix = pd.DataFrame(data=[first_row, second_row])

        return self.__transformation_matrix

    def __repr__(self):
        return f"Rotation2D(point={self.point}, angle={self.angle})"

    def __mul__(self, other: Rotation2D):
        return self.transformation_matrix.dot(other=other.transformation_matrix)
