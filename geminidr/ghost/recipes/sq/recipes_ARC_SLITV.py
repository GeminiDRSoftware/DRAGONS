"""
Recipes available to data with tags ``['GHOST', 'CAL', 'SLITV', 'ARC']``.
Default is ``makeProcessedSlitArc``, which is an alias to
:any:`makeProcessedSlit`.
"""
recipe_tags = set(['GHOST', 'CAL', 'SLITV', 'ARC'])

from .recipes_SLITV import makeProcessedSlit as makeProcessedSlitArc

_default = makeProcessedSlitArc
