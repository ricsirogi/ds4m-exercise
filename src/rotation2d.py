from __future__ import annotations

import numpy as np
import pandas as pd


class Rotation2D:
    def __init__(self, point: tuple[float, float] = (0.0, 0.0), angle: float = 0.0):
        if not (isinstance(point, tuple) and len(point) == 2):
            raise TypeError("Point must be a 2-element tuple.")
        if not isinstance(angle, (int, float)):
            raise TypeError("Angle must be a float or integer.")

        self.point = point
        self.angle = angle

        self.__transformation_matrix = None
        self.__inverse_transformation_matrix = None

    @property
    def transformation_matrix(self):
        first_row = [np.cos(self.angle), -np.sin(self.angle)]
        second_row = [np.sin(self.angle), np.cos(self.angle)]

        self.__transformation_matrix = pd.DataFrame(data=[first_row, second_row])

        return self.__transformation_matrix

    @property
    def inverse_transformation_matrix(self):
        first_row = [np.cos(-self.angle), -np.sin(-self.angle)]
        second_row = [np.sin(-self.angle), np.cos(-self.angle)]

        self.__inverse_transformation_matrix = pd.DataFrame(data=[first_row, second_row])

        return self.__inverse_transformation_matrix

    def __repr__(self):
        return f"Rotation2D(point={self.point}, angle={self.angle})"

    def __mul__(self, other: Rotation2D):
        return self.transformation_matrix.dot(other=other.transformation_matrix)
