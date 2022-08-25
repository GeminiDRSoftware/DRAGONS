"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'FLAT'}

from ..sq.recipes_FLAT_LS_SPECT import (makeProcessedFlatStack,
                                        makeProcessedFlatNoStack)

_default = makeProcessedFlatNoStack

