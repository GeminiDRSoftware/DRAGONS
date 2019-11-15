#!/usr/bin/env python
"""
Tests for GMOS Spect LS primitives.
"""

import pytest


# ToDo @bquint: These files are not used for now but I am keeping them for future regression tests
test_cases = [

    # GMOS-N B600 at 0.600 um ---
    ('GMOS/GN-2018A-Q-302-56', [
        'N20180304S0121.fits',  # Standard
        'N20180304S0122.fits',  # Standard
        'N20180304S0123.fits',  # Standard
        'N20180304S0124.fits',  # Standard
        'N20180304S0125.fits',  # Standard
        'N20180304S0126.fits',  # Standard
        'N20180304S0204.fits',  # Bias
        'N20180304S0205.fits',  # Bias
        'N20180304S0206.fits',  # Bias
        'N20180304S0207.fits',  # Bias
        'N20180304S0208.fits',  # Bias
        'N20180304S0122.fits',  # Flat
        'N20180304S0123.fits',  # Flat
        'N20180304S0126.fits',  # Flat
        'N20180302S0397.fits',  # Arc
    ]),

]


if __name__ == '__main__':
    pytest.main()
