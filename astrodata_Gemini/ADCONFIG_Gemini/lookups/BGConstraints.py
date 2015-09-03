# Gives BG band constraints in mags / square arcsec for given (GMOS) filter
# These are derived by Paul Hirst based on both 
# http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/optical-sky-background
# And the model sky spectra in the ITC.
# There are uncertainties and caveats with these numbers, and discrepancies between different sources of data and
# different assumptions. Predicting sky brightness is not an exact science. These are probably affected by systematics
# at the +/- 0.3 level in some cases, especially i and z.

#NOTE. Nelson Zarate: Added JHK filter for NIRI. The user of this dictionaty is
# primitives_qa.py/measureBG() at the GENERALPrimitives level. Otherwise we would need
# to create a NIRI measureBG() primitive just to handle this dictionary.

bgConstraints = {
'u':{20:21.66, 50:19.49, 80:17.48},
'g':{20:21.62, 50:20.68, 80:19.36},
'r':{20:21.33, 50:20.32, 80:19.34},
'i':{20:20.44, 50:19.97, 80:19.30},
'z':{20:19.51, 50:19.04, 80:18.37},
'J':{20:16.20, 50:16.20, 80:16.20},
'H':{20:13.80, 50:13.80, 80:13.80},
'K':{20:14.60, 50:14.60, 80:14.60},
'Ks':{20:14.60, 50:14.60, 80:14.60},
}
