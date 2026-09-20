import numpy as np
import pytest

from src.rotation2d import Rotation2D


class TestRotation2D:
    @pytest.fixture
    def default_rotation(self) -> Rotation2D:
        """
        Fixture providing a standard 90-degree rotation around the origin.

        :return Rotation2D rotation: A 90-degree rotation object centered at (0.0, 0.0).
        """
        return Rotation2D(point=(0.0, 0.0), angle=90.0)

    def test_default_initialization(self):
        """
        Tests that constructor default values work as expected.
        """
        rot = Rotation2D()
        assert rot.point == (0.0, 0.0)
        assert rot.angle_deg == 0.0

    def test_initialization(self, default_rotation: Rotation2D):
        """
        Tests that initialization attributes are set correctly.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        assert default_rotation.point == (0.0, 0.0)
        assert default_rotation.angle_deg == 90.0
        assert default_rotation.angle_rad == pytest.approx(np.pi / 2)

    def test_transformation_matrix_values(self, default_rotation: Rotation2D):
        """
        Tests matrix values for a known 90-degree rotation around the origin.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        matrix = default_rotation.transformation_matrix

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)

        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        np.testing.assert_allclose(matrix, expected, atol=1e-7)

    def test_transformation_matrix_identity(self):
        """
        Tests that a 0-degree rotation yields the identity matrix.
        """
        rot_zero = Rotation2D(point=(0.0, 0.0), angle=0.0)
        matrix = rot_zero.transformation_matrix

        expected = np.eye(3)
        np.testing.assert_allclose(matrix, expected, atol=1e-7)

    def test_inverse_transformation_matrix_values(self, default_rotation: Rotation2D):
        """
        Tests inverse matrix values for a known 90-degree rotation.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        inv_matrix = default_rotation.inverse_transformation_matrix

        assert isinstance(inv_matrix, np.ndarray)
        assert inv_matrix.shape == (3, 3)

        expected = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_allclose(inv_matrix, expected, atol=1e-7)

    def test_inverse_matrix_cancels_transformation_matrix(self, default_rotation: Rotation2D):
        """
        Tests that multiplying transformation matrix with its inverse yields identity.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        matrix = default_rotation.transformation_matrix
        inv_matrix = default_rotation.inverse_transformation_matrix

        product = matrix @ inv_matrix
        expected_identity = np.eye(3)

        np.testing.assert_allclose(product, expected_identity, atol=1e-7)

    def test_repr(self, default_rotation: Rotation2D):
        """
        Tests string representation of the object.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        assert repr(default_rotation) == "Rotation(point=(0.0, 0.0), angle=90.0)"

    def test_multiplication_combines_transformations(self):
        """
        Tests multiplying two rotations around the same center combines their angles.
        """
        rot1 = Rotation2D(point=(1.0, 1.0), angle=45.0)
        rot2 = Rotation2D(point=(1.0, 1.0), angle=45.0)

        result_rot = rot1 * rot2

        assert isinstance(result_rot, Rotation2D)
        assert result_rot.point == (1.0, 1.0)
        assert result_rot.angle_deg == 90.0

    def test_multiplication_different_points_raises_value_error(self):
        """
        Tests that multiplying rotations around different points raises ValueError.
        """
        rot1 = Rotation2D(point=(0.0, 0.0), angle=45.0)
        rot2 = Rotation2D(point=(1.0, 1.0), angle=45.0)

        with pytest.raises(ValueError):
            _ = rot1 * rot2

    def test_invalid_multiplication_raises_type_error(self, default_rotation: Rotation2D):
        """
        Tests that multiplying with an unsupported type raises TypeError.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """
        with pytest.raises(TypeError):
            _ = default_rotation * "not_a_rotation_object"
