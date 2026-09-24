from __future__ import annotations

import numpy as np


class Rotation2D:
    def __init__(self, point: tuple[float | int, float | int] = (0.0, 0.0), angle: float | int = 0.0):
        """
        Initializes a 2D rotation around a center point by a given angle in degrees.

        :param tuple[float | int, float | int] point: The center point of rotation (x, y). Defaults to (0.0, 0.0).
        :param float | int angle: The rotation angle in degrees. Defaults to 0.0.
        """

        Rotation2D.check_input(angle=angle, point=point)

        self.point: tuple[float | int, float | int] = (point[0], point[1])
        self.angle_rad: np.number = np.deg2rad(angle)
        self.angle_deg: float | int = angle

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

    def __call__(self, point_to_rotate: tuple[float | int, float | int]) -> tuple[float | int, float | int]:
        """
        Rotates the point given around the center point by the angle of the rotation.

        :param tuple[float | int, float | int] point_to_rotate: The point to be rotated
        :return tuple[float | int, float | int]: The rotated point.
        """
        Rotation2D.check_input(point=point_to_rotate)

        new_point = np.array(object=[point_to_rotate[0], point_to_rotate[1], 1])

        result = Rotation2D.get_rotation_matrix(self.angle_rad, point=self.point) @ new_point

        return float(result[0]), float(result[1])

    @property
    def inverse_transformation_matrix(self):
        """
        Gets the inverse transformation matrix for this rotation.

        :return np.ndarray __inverse_transformation_matrix: The 3x3 homogeneous matrix representing the inverse rotation.
        """

        self.__inverse_transformation_matrix = Rotation2D.get_rotation_matrix(-self.angle_rad, point=self.point)

        return self.__inverse_transformation_matrix

    @property
    def transformation_matrix(self):
        """
        Gets the transformation matrix associated with this rotation.

        :return np.ndarray __transformation_matrix: The 3x3 homogeneous matrix representing the forward rotation.
        """

        self.__transformation_matrix = Rotation2D.get_rotation_matrix(self.angle_rad, point=self.point)

        return self.__transformation_matrix

    @staticmethod
    def check_input(angle: float | int | np.number = None,
                    point: tuple[float | int, float | int] = None):

        if angle is not None:
            if not isinstance(angle, (int, float, np.number)):
                raise TypeError("Angle must be a float or integer.")
        if point is not None:
            if not (isinstance(point, (tuple, list)) and len(point) == 2):
                raise TypeError("Point must be a 2-element tuple or list.")
            if not all(isinstance(x, (int, float, np.number)) for x in point):
                raise TypeError("Point elements must be numeric.")

    @staticmethod
    def get_rotation_matrix(angle: float | int | np.number, point: tuple[float | int, float | int]) -> np.ndarray:
        """
        Calculates the 3x3 homogeneous transformation matrix for a 2D rotation.

        :param float | int angle: The rotation angle in radians.
        :param tuple[float | int, float | int] point: The center point of rotation (x, y).

        :return np.ndarray matrix: The 3x3 transformation matrix.
        """
        Rotation2D.check_input(angle=angle, point=point)

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
