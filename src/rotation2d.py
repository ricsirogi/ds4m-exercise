from __future__ import annotations

import numpy as np


class Rotation2D:
    def __init__(self, point: tuple[float, float] = (0.0, 0.0), angle: float = 0.0):
        """
        Initializes a 2D rotation around a center point by a given angle in degrees.

        :param tuple[float, float] point: The center point of rotation (x, y). Defaults to (0.0, 0.0).
        :param float angle: The rotation angle in degrees. Defaults to 0.0.
        """

        if not (isinstance(point, tuple) and len(point) == 2):
            raise TypeError("Point must be a 2-element tuple.")
        if not isinstance(angle, (int, float)):
            raise TypeError("Angle must be a float or integer.")

        self.point = point
        self.angle_rad = np.deg2rad(angle)
        self.angle_deg = angle

        self.__transformation_matrix = None
        self.__inverse_transformation_matrix = None

    def __repr__(self):
        """
        Returns a string representation of the Rotation2D instance.

        :return str repr: Formatted string representing the object.
        """

        return f"Rotation(point={self.point}, angle={self.angle_deg})"

    def __mul__(self, other: Rotation2D):
        """
        Composes two rotations that share the same center point.

        :param Rotation2D other: Another rotation instance to compose with.

        :return Rotation2D result: A new rotation instance representing the combined rotation.
        """

        if not isinstance(other, Rotation2D):
            return NotImplemented
        if other.point == self.point:
            return Rotation2D(point=self.point, angle=self.angle_deg + other.angle_deg)
        else:
            raise ValueError("Rotation points don't match")

    @property
    def inverse_transformation_matrix(self):
        """
        Gets the inverse transformation matrix for this rotation.

        :return np.ndarray matrix: The 3x3 homogeneous matrix representing the inverse rotation.
        """

        self.__inverse_transformation_matrix = Rotation2D.get_rotation_matrix(-self.angle_rad, point=self.point)

        return self.__inverse_transformation_matrix

    @property
    def transformation_matrix(self):
        """
        Gets the transformation matrix associated with this rotation.

        :return np.ndarray matrix: The 3x3 homogeneous matrix representing the forward rotation.
        """

        self.__transformation_matrix = Rotation2D.get_rotation_matrix(self.angle_rad, point=self.point)

        return self.__transformation_matrix

    @staticmethod
    def get_rotation_matrix(angle: float, point: tuple[float, float]) -> np.ndarray:
        """
        Calculates the 3x3 homogeneous transformation matrix for a 2D rotation.

        :param float angle: The rotation angle in radians.
        :param tuple[float, float] point: The center point of rotation (x, y).

        :return np.ndarray matrix: The 3x3 transformation matrix.
        """

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
