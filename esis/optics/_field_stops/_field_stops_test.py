import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import esis


class AbstractTestAbstractFieldStop(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestReplaceable,
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_num_folds(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.num_folds
        assert isinstance(result, int)
        assert result >= 0

    def test_num_sides(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.num_sides
        assert isinstance(result, int)
        assert result >= 0

    def test_radius_clear(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.radius_clear
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_width_clear(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.width_clear
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_radius_mechanical(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.radius_mechanical
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_vertices(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.vertices
        if result is not None:
            assert isinstance(result, na.AbstractCartesian2dVectorArray)
            assert "vertex" in result.shape
            assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_aperture(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.aperture
        assert isinstance(result, optika.apertures.AbstractPolygonalAperture)
        if a.vertices is not None:
            num = a.vertices.shape["vertex"]
            assert result.vertices.shape["vertex"] == num

    def test_surface(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert result.is_field_stop
        assert result.aperture is not None
        assert result.aperture_mechanical is not None


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.FieldStop(),
        esis.optics.FieldStop(
            num_folds=8,
            radius_clear=5 * u.mm,
            radius_mechanical=10 * u.mm,
            translation=na.Cartesian3dVectorArray(z=200) * u.mm,
        ),
        esis.optics.FieldStop(
            num_folds=8,
            radius_clear=5 * u.mm,
            radius_mechanical=10 * u.mm,
            vertices=na.Cartesian2dVectorArray(
                x=na.ScalarArray(
                    5 * np.cos(np.arange(8) * np.pi / 4) * u.mm, axes="vertex"
                ),
                y=na.ScalarArray(
                    4.9 * np.sin(np.arange(8) * np.pi / 4) * u.mm, axes="vertex"
                ),
            ),
        ),
    ],
)
class TestFieldStop(
    AbstractTestAbstractFieldStop,
):
    pass
