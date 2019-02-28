import __future__

import pytest
import os
import numpy as np
import astrodata
import gemini_instruments
from gempy.utils import logutils

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')

# Each class has it's own fixture
@pytest.fixture(scope='module')
def setup_module(request):
    print('setup reduce_img_nodarks module')

    def fin():
        print('\nteardown reduce_img_nodarks module')
    request.addfinalizer(fin)
    return

@pytest.fixture(scope='class')
def setup_testimgnodarks(request):
    print('setup testimgnodarks')

    def fin():
        print('\nteardown testimgnodarks')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_testimgnodarks')
class TestImgNoDarks():
    pass