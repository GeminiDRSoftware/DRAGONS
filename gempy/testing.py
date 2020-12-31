import os
import pytest

from astropy.modeling import models

from gempy.library.astromodels import get_named_submodel
from numpy.testing import assert_allclose


def assert_have_same_distortion(ad, ad_ref, atol=0, rtol=1e-7):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    same distortion.

    Parameters
    ----------
    ad : :class:`astrodata.AstroData`
        AstroData object to be checked.
    ad_ref : :class:`astrodata.AstroData`
        AstroData object used as reference
    atol, rtol : float
        absolute and relative tolerances
    """
    for ext, ext_ref in zip(ad, ad_ref):
        m = ext.wcs.get_transform(ext.wcs.input_frame, "distortion_corrected")
        m_ref = ext_ref.wcs.get_transform(ext_ref.wcs.input_frame,
                                          "distortion_corrected")

        # The [1] index is because the transform is always
        # Mapping | Chebyshev2D & Identity(1)
        assert m[1].__class__.__name__ == m_ref[1].__class__.__name__ == "Chebyshev2D"
        assert_allclose(m[1].parameters, m_ref[1].parameters,
                        atol=atol, rtol=rtol)
        assert_allclose(m.inverse[1].parameters, m_ref.inverse[1].parameters,
                        atol=atol, rtol=rtol)


def assert_wavelength_solutions_are_close(ad, ad_ref, atol=0, rtol=1e-7):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    wavelength solution.

    Parameters
    ----------
    ad : :class:`astrodata.AstroData` or any subclass
        AstroData object to be checked.
    ad_ref : :class:`astrodata.AstroData` or any subclass
        AstroData object used as reference
    atol, rtol : float
        absolute and relative tolerances
    """
    for ext, ext_ref in zip(ad, ad_ref):
        wcal = get_named_submodel(ext.wcs.forward_transform, 'WAVE')
        wcal_ref = get_named_submodel(ext_ref.wcs.forward_transform, 'WAVE')
        assert_allclose(wcal.parameters, wcal_ref.parameters,
                        atol=atol, rtol=rtol)


def dict_to_polynomial(model_dict):
    """
    This is the inverse of polynomial_to_dict(), taking a dict of property/
    parameter names and their values and making a suitable model instance.

    Parameters
    ----------
    model_dict: dict
        Dictionary with pair/value that defines the Chebyshev model.

    Returns
    -------
    models.ChebyshevND or None
        Returns the models if it is parsed successfully. If not, it will return
        None.
    """
    try:
        model_class = model_dict.pop("model")
        ndim = int(model_class[-2])
    except KeyError:  # Handle old models (assumed to be Chebyshevs)
        try:
            ndim = int(model_dict.pop("ndim"))
        except KeyError:
            return None
        model_class = f"Chebyshev{ndim}D"
    if "ndim" in model_dict:
        del model_dict["ndim"]

    cls = getattr(models, model_class)

    if ndim == 1:
        model = cls(degree=int(model_dict.pop("degree")))
    elif ndim == 2:
        model = cls(x_degree=int(model_dict.pop("x_degree")),
                    y_degree=int(model_dict.pop("y_degree")))

    model.meta["xunit"] = model_dict.pop("xunit", None)
    model.meta["yunit"] = model_dict.pop("yunit", None)

    for k, v in model_dict.items():
        try:
            if k.endswith("domain_start"):
                setattr(model, k.replace("_start", ""),
                        [float(v), float(model_dict[k.replace("start", "end")])])
            elif k and not k.endswith("domain_end"):  # ignore k==""
                setattr(model, k, float(v))
        except (KeyError, AttributeError):
            return None

    return model
