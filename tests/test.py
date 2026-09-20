import numpy as np
import pandas as pd
import pytest

from src.rotation2d import Rotation2D  # Adjust the import based on your module name


class TestRotation2D:
    @pytest.fixture
    def default_rotation(self) -> Rotation2D:
        """
        Fixture providing a standard 90-degree (pi/2) rotation around the origin.
        """

        return Rotation2D(point=(0.0, 0.0), angle=np.pi / 2)

    def test_initialization(self, default_rotation: Rotation2D):
        """
        Test that initialization attributes are set correctly.

        :param Rotation2D default_rotation: The rotation object used for testing
        """

        assert default_rotation.point == (0.0, 0.0)
        assert default_rotation.angle == pytest.approx(np.pi / 2)

    def test_transformation_matrix_values(self, default_rotation: Rotation2D):
        """
        Test matrix values for a known 90-degree rotation.

        :param Rotation2D default_rotation: The rotation object used for testing
        """

        matrix = default_rotation.transformation_matrix

        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (2, 2)

        # cos(pi/2) ≈ 0, sin(pi/2) = 1
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(matrix.values, expected, atol=1e-7)

    def test_transformation_matrix_identity(self):
        """
        Test that a 0-degree rotation yields the identity matrix.
        """

        rot_zero = Rotation2D(point=(0.0, 0.0), angle=0.0)
        matrix = rot_zero.transformation_matrix

        expected = np.eye(2)
        np.testing.assert_allclose(matrix.values, expected, atol=1e-7)

    def test_repr(self, default_rotation: Rotation2D):
        """
        Test string representation of the object.

        :param Rotation2D default_rotation: The rotation object used for testing
        """

        assert repr(default_rotation) == f"Rotation2D(point=(0.0, 0.0), angle={np.pi / 2})"

    def test_multiplication_combines_transformations(self):
        """
        Test multiplying two rotations yields the expected matrix multiplication result.
        """

        rot1 = Rotation2D(point=(0.0, 0.0), angle=np.pi / 4)  # 45 deg
        rot2 = Rotation2D(point=(0.0, 0.0), angle=np.pi / 4)  # 45 deg

        # Multiplying two 45-degree rotations should equal a 90-degree rotation matrix
        result_df = rot1 * rot2
        expected_90_deg = Rotation2D(point=(0.0, 0.0), angle=np.pi / 2).transformation_matrix

        assert isinstance(result_df, pd.DataFrame)
        np.testing.assert_allclose(result_df.values, expected_90_deg.values, atol=1e-7)

    def test_invalid_multiplication_raises(self):
        """
        Test that multiplying with an object lacking transformation_matrix fails.
        """

        rot = Rotation2D(point=(0.0, 0.0), angle=0.0)
        with pytest.raises(AttributeError):
            _ = rot * "not_a_rotation_object"
