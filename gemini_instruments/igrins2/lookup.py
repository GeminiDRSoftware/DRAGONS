import numpy as np

# If the filter names and central wavelength are different from
# the global definitions in gemini_instruments/gemini/lookup.py
# redefine them here in filter_wavelengths.

filter_wavelengths = {
}

array_properties = {
    # EDIT AS NEEDED
    # "gain"  :  3,   # electrons/ADU  (MADE UP VALUE for example)
    "read_noise_fit": {"H": [25.6, 3],
                       "K": [25.6, 3]},
}
