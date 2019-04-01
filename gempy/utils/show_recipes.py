#!/usr/bin/env python
# -*- coding: utf8 -*-=
import os
import sys
import inspect
import importlib

import geminidr
import astrodata
import gemini_instruments
from astrodata.core import AstroDataError


from recipe_system.mappers.recipeMapper import RecipeMapper


def show_recipes(_file):
    """
    show_recipes takes in a file, and will return all the possible recipes
    that can be used on the file.

    Parameters
    ----------
    file - str
        file name for the fits file

    Returns - None
    Prints out information
    -------
    """

    # Find the file and open it with astrodata
    try:
        ad = astrodata.open(_file)
        tags = ad.tags
    except AstroDataError:
        print("There was an issue using the selected file, please check"
              "the format and directory:", sys.exc_info()[0])
        raise

    # a list of current instruments needs to be found so show_recipies
    # can parse through those directories to find the recipes. Can't
    # assume static list, user may add instrument
    list_of_found_instruments = []

    # will return the location of dragons, as ".../dragons/"
    local_dir = '/'.join(geminidr.__file__.split("/")[:-2]) + '/'

    # returns every folder, including all subfolders which need to be parsed
    all_folders = [x[0] for x in os.walk(
        os.path.expanduser(local_dir + 'geminidr/'))]

    for i in all_folders:

        # 6th element of directory is where any folder under /geminidr is
        instrument_name = (i.split("/")[6]).lower()

        # Folders in the same directory, but are known not to be instruments
        not_instruments = ['doc', '', 'core', '__pycache__', 'gemini']

        # If instrument_name has not been added to the list,
        # and isn't in not_instruments, add it
        if instrument_name not in list_of_found_instruments:
            if instrument_name not in not_instruments:
                list_of_found_instruments.append(instrument_name)

    # Tests to make sure an instrument was found
    intersect_string = \
        " The instrument in the file provided did not match any of" \
        " the know instruments in the /geminidr directory. All \n" \
        " recipes exist in this directory, and no folder was" \
        " associated with the name of the instrument provided.\n" \
        " Check to see if the file provided has an instrument" \
        " associated with it, and that the instrument exists in /geminidr." \
        " \n The instrument was found to be {}, and the tags " \
        "were {}".format(ad.instrument(), tags)

    instrument = ad.instrument().lower()

    if instrument in ["gmos-s", "gmos-n"]:
        instrument = "gmos"

    assert instrument in list_of_found_instruments, intersect_string

    # Finds of the file is DARK, FLAT, BIAS, NS or IMAGE so import_module
    # can import the correct module to obtain the proper recipe

    # This will be cleaned up!
    if "DARK" in tags:
        tag_object = "DARK"
        module = 'recipes_DARK'
    elif "FLAT" in tags:
        tag_object = "FLAT"
        module = 'recipes_FLAT_IMAGE'
    elif "BIAS" in tags:
        tag_object = "BIAS"
        module = 'recipes_BIAS'
    elif "NODANDSHUFFLE" in tags:
        tag_object = "NODANDSHUFFLE"
        module = 'recipes_NS'
    else:
        tag_object = "IMAGE"
        module = 'recipes_IMAGE'

    all_recipies = []

    for mode in ['sq', 'qa', 'ql']:

        # sets up full path to import, dynamically finds  characteristics
        absolute_dir = 'geminidr.' + instrument + '.recipes.' \
                       + mode + '.' + module

        # Makes sure the discovered path where the recipe is stores exists
        absolute_path = absolute_dir.replace(".", "/")
        exp_usr = os.path.expanduser(local_dir + absolute_path + ".py")

        if os.path.exists(exp_usr):
            # creates the import statement of the module that is needed
            mod = importlib.import_module(absolute_dir)

            # finds all functions(recipies) in the module except for 'default'
            functions_list = [i[0] for i in inspect.getmembers(mod)
                              if inspect.isfunction(i[1]) and (i[0] != 'default')]

            # Appends said recipes to an external list so it can be called later
            for i in range(len(functions_list)):
                all_recipies.append(absolute_dir + "::" + functions_list[i])

    # Output
    print("Input file: .{}".format(_file))
    print("Input tags: {}".format(tags))
    print("Recipes available for the input file: ")
    if all_recipies == []:
        print("No recipes were found for this file!")
    for recipe in all_recipies:
        print("  " + recipe)
    return
