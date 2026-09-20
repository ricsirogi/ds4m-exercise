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
        self.angle_rad = np.deg2rad(angle)
        self.angle_deg = angle

        self.__transformation_matrix = None
        self.__inverse_transformation_matrix = None

    @property
    def transformation_matrix(self):
        self.__transformation_matrix = Rotation2D.get_rotation_matrix(self.angle_rad, point=self.point)

        return self.__transformation_matrix

    @staticmethod
    def get_rotation_matrix(angle: float, point: tuple[float, float]) -> np.ndarray:
        first_matrix = np.array([[1, 0, point[0]],
                                   [0, 1, point[1]],
                                   [0, 0, 1]])

        second_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]])

        third_matrix = np.array([[1, 0, -point[0]],
                                 [0, 1, -point[1]],
                                 [0, 0, 1]])

        return first_matrix @ second_matrix @ third_matrix

    @property
    def inverse_transformation_matrix(self):
        self.__inverse_transformation_matrix = Rotation2D.get_rotation_matrix(-self.angle_rad, point=self.point)

        return self.__inverse_transformation_matrix

    def __repr__(self):
        return f"Rotation(point={self.point}, angle={self.angle_deg})"

    def __mul__(self, other: Rotation2D):
        return self.transformation_matrix.dot(other=other.transformation_matrix)
