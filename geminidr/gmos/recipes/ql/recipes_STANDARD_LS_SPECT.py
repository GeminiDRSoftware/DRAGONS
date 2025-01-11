"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'STANDARD'].
These are GMOS longslit arc-lamp calibrations.
Default is "reduceStandard".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'STANDARD'}

from geminidr.gmos.recipes.ql.recipes_LS_SPECT import reduceScience, reduceStandard
from geminidr.gmos.recipes.ql.recipes_common import makeIRAFCompatible

_default = reduceStandard
