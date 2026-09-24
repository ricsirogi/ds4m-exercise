import numpy as np
import pytest

from src.rotation2d import Rotation2D


class TestRotation2D:
    """
    Test suite for the Rotation2D class.
    """

    @pytest.fixture
    def default_rotation(self) -> Rotation2D:
        """
        Fixture providing a standard 90-degree rotation around the origin.

        :return Rotation2D rotation: A 90-degree rotation object centered at (0.0, 0.0).
        """

        return Rotation2D(point=(0.0, 0.0), angle=90.0)

    # -------------------------------------------------------------------------
    # Initialization & Attributes
    # -------------------------------------------------------------------------

    def test_default_initialization(self):
        """
        Tests that constructor default values work as expected.
        """

        rot = Rotation2D()
        assert rot.point == (0.0, 0.0)
        assert rot.angle_deg == 0.0

    def test_initialization_with_list(self):
        """
        Tests that passing a list converts point to a tuple.
        """

        rot = Rotation2D(point=[1, 2], angle=45)
        assert rot.point == (1.0, 2.0)
        assert isinstance(rot.point, tuple)

    def test_initialization_attributes(self, default_rotation: Rotation2D):
        """
        Tests that initialization attributes are set correctly.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """

        assert default_rotation.point == (0.0, 0.0)
        assert default_rotation.angle_deg == 90.0
        assert default_rotation.angle_rad == pytest.approx(np.pi / 2)

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def test_repr(self, default_rotation: Rotation2D):
        """
        Tests string representation of the object.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """

        assert repr(default_rotation) == "Rotation(point=(0.0, 0.0), angle=90.0)"

    # -------------------------------------------------------------------------
    # Point Rotation Execution (__call__)
    # -------------------------------------------------------------------------

    def test_call_origin(self, default_rotation: Rotation2D):
        """
        Tests executing rotation on a point around the origin using __call__.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """

        transformed_point = default_rotation((1.0, 0.0))
        assert transformed_point == pytest.approx((0.0, 1.0), abs=1e-7)

    def test_call_non_origin_center(self):
        """
        Tests executing rotation on a point around a non-origin center using __call__.
        """

        rot = Rotation2D(point=(2.0, 2.0), angle=180.0)
        transformed_point = rot((3.0, 2.0))
        assert transformed_point == pytest.approx((1.0, 2.0), abs=1e-7)

    def test_call_accepts_list_input(self, default_rotation: Rotation2D):
        """
        Tests executing __call__ with a list instead of a tuple.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """

        transformed_point = default_rotation([1.0, 0.0])
        assert transformed_point == pytest.approx((0.0, 1.0), abs=1e-7)

    # -------------------------------------------------------------------------
    # Matrix Calculations & Properties
    # -------------------------------------------------------------------------

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
            [ 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [ 0.0, 0.0, 1.0]
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

    def test_get_rotation_matrix_static_method(self):
        """
        Tests calling the get_rotation_matrix static method directly.
        """

        matrix = Rotation2D.get_rotation_matrix(np.pi / 2, point=(0.0, 0.0))
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        np.testing.assert_allclose(matrix, expected, atol=1e-7)

    # -------------------------------------------------------------------------
    # Composition (__mul__)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Validation & Error Handling
    # -------------------------------------------------------------------------

    def test_invalid_input_types_raise_type_error(self, default_rotation: Rotation2D):
        """
        Tests that passing invalid input types or non-numeric elements raises TypeError.

        :param Rotation2D default_rotation: The rotation fixture object used for testing.
        """

        with pytest.raises(TypeError):
            Rotation2D(point=("invalid", "type"), angle=90.0)

        with pytest.raises(TypeError):
            Rotation2D(point=(0.0, 0.0), angle="invalid")

        with pytest.raises(TypeError):
            default_rotation("not_a_point")

        with pytest.raises(TypeError):
            default_rotation(("a", "b"))

    def test_invalid_point_lengths_raise_type_error(self):
        """
        Tests that passing points with length != 2 raises TypeError.
        """

        with pytest.raises(TypeError):
            Rotation2D(point=(1.0, 2.0, 3.0))

        with pytest.raises(TypeError):
            Rotation2D(point=(1.0,))

        with pytest.raises(TypeError):
            Rotation2D(point=())

    def test_get_rotation_matrix_invalid_inputs_raise_type_error(self):
        """
        Tests that static method get_rotation_matrix validates inputs independently.
        """

        with pytest.raises(TypeError):
            Rotation2D.get_rotation_matrix("invalid_angle", point=(0.0, 0.0))

        with pytest.raises(TypeError):
            Rotation2D.get_rotation_matrix(np.pi, point="invalid_point")
