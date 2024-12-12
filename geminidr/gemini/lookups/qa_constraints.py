"""Gives BG band constraints in mags / square arcsec for given (GMOS) filter
These are derived by Paul Hirst based on both
http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/optical-sky-background
And the model sky spectra in the ITC.
There are uncertainties and caveats with these numbers, and discrepancies between different sources of data and
different assumptions. Predicting sky brightness is not an exact science. These are probably affected by systematics
at the +/- 0.3 level in some cases, especially i and z.
"""

# NOTE. Nelson Zarate: Added JHK filter for NIRI. The user of this dictionaty is
# primitives_qa.py/measureBG() at the GENERALPrimitives level. Otherwise we would need
# to create a NIRI measureBG() primitive just to handle this dictionary.

bgBands = {
    "u": {20: 21.66, 50: 19.49, 80: 17.48},
    "g": {20: 21.62, 50: 20.68, 80: 19.36},
    "r": {20: 21.33, 50: 20.32, 80: 19.34},
    "i": {20: 20.44, 50: 19.97, 80: 19.30},
    "z": {20: 19.51, 50: 19.04, 80: 18.37},
    "J": {20: 16.20, 50: 16.20, 80: 16.20},
    "H": {20: 13.80, 50: 13.80, 80: 13.80},
    "K": {20: 14.60, 50: 14.60, 80: 14.60},
    "Ks": {20: 14.60, 50: 14.60, 80: 14.60},
}

# Gives CC band constraints as a function of magnitudes of extinction.
# From http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints
# the value defines the maximum amount of extinction that classes as that band.

# Photometry to better than 5% (0.05 mags) is difficult due to scattered light from the
# focal reducer,  also we're using a nominal rather than measured atmospheric extinction
# so we take the photometric number to be less than 0.08 mags extinction.

ccBands = {50: 0.08, 70: 0.3, 80: 1.0}

# Gives IQ band constraints for given filter + wavefront sensor combination
# Note that X has just been set to the average of Y and J.
iqBands = {
    "u": {20: 0.60, 70: 0.90, 85: 1.20},
    "g": {20: 0.60, 70: 0.85, 85: 1.10},
    "r": {20: 0.50, 70: 0.75, 85: 1.05},
    "i": {20: 0.50, 70: 0.75, 85: 1.05},
    "Z": {20: 0.50, 70: 0.70, 85: 0.95},
    "Y": {20: 0.40, 70: 0.70, 85: 0.95},
    "X": {20: 0.40, 70: 0.65, 85: 0.90},
    "J": {20: 0.40, 70: 0.60, 85: 0.85},
    "H": {20: 0.40, 70: 0.60, 85: 0.85},
    "K": {20: 0.35, 70: 0.55, 85: 0.80},
    "L": {20: 0.35, 70: 0.50, 85: 0.75},
    "M": {20: 0.35, 70: 0.50, 85: 0.70},
    "N": {20: 0.34, 70: 0.37, 85: 0.45},
    "Q": {20: 0.54, 70: 0.54, 85: 0.54},
    "AO": {20: 0.45, 70: 0.80, 85: 1.20},  # from view_wfs.py (telops ~/bin)
}

# Gives WV band constraints (in mm of precipitable H2O at zenith) for Mauna Kea and Cerro Pachon.
wvBands = {
    "Gemini-North": {"20": 1.0, "50": 1.6, "80": 3.0},
    "Gemini-South": {"20": 2.3, "50": 4.3, "80": 7.6},
}
