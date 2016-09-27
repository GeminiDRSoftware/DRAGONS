# Gives CC band constraints as a function of magnitudes of extinction.
# From http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints
# the value defines the maximum amount of extinction that classes as that band.

# Photometry to better than 5% (0.05 mags) is difficult due to scattered light from the 
# focal reducer,  also we're using a nominal rather than measured atmospheric extinction
# so we take the photometric number to be less than 0.08 mags extinction.

ccConstraints = {
'50':0.08, '70':0.3, '80':1.0
}
