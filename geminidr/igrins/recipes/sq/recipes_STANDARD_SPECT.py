"""
Recipes available to data with tags ['IGRINS-2', 'SPECT', 'XD'', 'STANDARD].
Default is "reduceTelluric".
"""
from .recipes_SPECT import reduceTelluric, reduceScience, makeArcFromScience


recipe_tags = {'IGRINS-2', 'SPECT', 'XD', 'STANDARD'}

_default = reduceTelluric
