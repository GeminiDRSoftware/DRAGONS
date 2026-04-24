# import astropy.io.fits as pyfits
import numpy as np
import astrodata, gemini_instruments


from recipe_system.mappers.recipeMapper import RecipeMapper
from recipe_system.mappers.primitiveMapper import PrimitiveMapper


def get_pclass_and_recipe(ad):
    pm = PrimitiveMapper(ad.tags)
    pclass = pm.get_applicable_primitives()

    rmapper = RecipeMapper(ad.tags, recipename="makeProcessedBPM")
    recipe = rmapper.get_applicable_recipe()

    return pclass, recipe

from pathlib import Path

fn = "N20240429S0204_K.fits"
fn = "N20240429S0204_H.fits"
ad = astrodata.open(fn)
adlist = [ad]

data_list = [ad[0].data for ad in adlist]

ext = ad[0]
band = ext.band()

from geminidr.igrins2.procedures.readout_pattern.readout_pattern_helper import pipes, apply_pipe

p = [pipes["p64_global_median"]]
data_list2 = [apply_pipe(d, p) for d in data_list]

