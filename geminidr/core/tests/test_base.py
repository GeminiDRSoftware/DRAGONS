# Tests associated with PrimitivesBASE
import pytest

from geminidr.f2.primitives_f2_image import F2Image
from geminidr.gmos.primitives_gmos_image import GMOSImage


CORRECT_PARAMETERS = [('GMOSImage', 'standardizeWCS', ['suffix']),
                      ('GMOSImage', 'prepare', ['suffix', 'attach_mdf', 'mdf', 'require_wcs']),
                      ('F2Image', 'stackDarks',
                       ['suffix', 'apply_dq', 'statsec', 'operation',
                        'reject_method', 'hsigma', 'lsigma', 'mclip',
                        'max_iters', 'nlow', 'nhigh', 'memory',
                        'save_rejection_map', 'separate_ext', 'debug_pixel']),
                      ]


@pytest.mark.parametrize("cls,primitive,parameters", CORRECT_PARAMETERS)
def test_parameter_inheritance(cls, primitive, parameters):
    """Tests to ensure that primitives have the correct set of parameters"""
    p = globals()[cls]([])
    params = p.params[primitive].keys()
    assert set(params) == set(parameters)
