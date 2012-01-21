# Gives BG band constraints in mags / square arcsec for given (GMOS) filter
# These are derived by Paul Hirst based on both 
# http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/optical-sky-background
# And the model sky spectra in the ITC.
# There are uncertainties and caveats with these numbers, and discrepancies between different sources of data and
# different assumptions. Predicting sky brightness is not an exact science. These are probably affected by systematics
# at the +/- 0.3 level in some cases, especially i and z.

bgConstraints = {
'u':{20:21.66, 50:19.49, 80:17.48},
'g':{20:21.62, 50:20.68, 80:19.36},
'r':{20:21.33, 50:20.32, 80:19.34},
'i':{20:20.44, 50:19.97, 80:19.30},
'z':{20:19.47, 50:19.42, 80:19.33},
}
