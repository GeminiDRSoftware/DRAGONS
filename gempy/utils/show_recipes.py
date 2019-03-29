#!/usr/bin/env python
# -*- coding: utf8 -*-


import inspect
import astrodata
import gemini_instruments
import os
import importlib
import sys


def show_recipes(_file):
    """
    show_recipes takes in a file, and will return all the possible recipes that can
    be used on the file.

    Parameters
    ----------
    file - str
        file name for the fits file

    Returns - multiple string lines
    -------
    """

    # Find the file and open it with astrodata
    try:
        if os.path.isabs(_file):
            ad = astrodata.open(_file)
            assert ad.tags
        elif not os.path.isabs(_file):
            ad = astrodata.open(os.path.join(os.getcwd(), _file))
        else:
            raise OSError("Could not find file. The file provided was neither "
                          "an absolute file location or part of the current \n"
                          "working directory. Please check if file exists in "
                          "path provided")
    except:
        print("There was an issue using the selected file, please check"
              "the format and directory:", sys.exc_info()[0])
        raise

    # a list of current instruments needs to be found so show_recipies can parse through
    # those directories to find the recipes. Can't assume static list, user may add instrument
    list_of_found_instruments = []

    # returns every folder, including all subfolders which need to be parsed out
    all_folders = [x[0] for x in os.walk(os.path.expanduser("~/workspace/dragons/geminidr/"))]

    for i in all_folders:

        # Returns the 6th element of the directory, which is any folder right under /geminidr
        instrument_name = (i.split("/")[6]).lower()

        # The following are folders in the same directory, but are known not to be instruments
        not_instruments = ['doc', '', 'core', '__pycache__', 'gemini']

        # If instrument_name has not been added to the list,
        # and isn't in not_instruments, add it
        if instrument_name not in list_of_found_instruments:
            if instrument_name not in not_instruments:
                list_of_found_instruments.append(instrument_name)

    # Tests to make sure an instrument was found
    intersect_string = " The instrument in the file provided did not match any of" \
                       " the know instruments in the /geminidr directory. All \n" \
                       " recipes exist in this directory, and no folder was" \
                       " associated with the name of the instrument provided.\n" \
                       " Check to see if the file provided has an instrument" \
                       " associated with it, and that the instrument exists in /geminidr." \
                       " \n The instrument was found to be {}, and the tags " \
                       "were {}".format(ad.instrument(), ad.tags)

    instrument = ad.instrument().lower()

    if instrument in ["gmos-s", "gmos-n"]:
        instrument = "gmos"

    assert instrument in list_of_found_instruments, intersect_string

    # Finds of the file is DARK, FLAT, BIAS, NS or IMAGE so import_module can import
    # the correct module to obtain the proper recipe
    if "DARK" in ad.tags:
        tag_object = "DARK"
        module = 'recipes_DARK'
    elif "FLAT" in ad.tags:
        tag_object = "FLAT"
        module = 'recipes_FLAT_IMAGE'
    elif "BIAS" in ad.tags:
        tag_object = "BIAS"
        module = 'recipes_BIAS'
    elif "NODANDSHUFFLE" in ad.tags:
        tag_object = "NODANDSHUFFLE"
        module = 'recipes_NS'
    else:
        tag_object = "IMAGE"
        module = 'recipes_IMAGE'

    all_recipies = []

    for mode in ['sq', 'qa']:

        # sets up full path to import, dynamically finds instrument, mode, and module characteristics
        absolute_dir = 'geminidr.' + instrument + '.recipes.' + mode + '.' + module

        # Tests to make sure the discovered path where the recipe is stores exists
        absolute_path = absolute_dir.replace(".", "/")
        exp_usr = os.path.expanduser("~/workspace/dragons/" + absolute_path + ".py")

        try:
            assert (os.path.exists(exp_usr))
        except AssertionError:
            raise Warning("The expected location of the recipe does not exist. "
                          "'show_recipes' found the instrument to be '{}', the mode \n"
                          "to be '{}', and the object to be '{}', but could not find a"
                          " module in the expected directory: '{}.recipes.{}.{}.py'. \n"
                          "This may mean that the given file lacks a '{}' mode or that there "
                          "no recipes for either the instrument '{}', or the objecttype '{}'"
                          .format(instrument, mode.lower(), tag_object,
                                  instrument, mode.lower(), module,
                                  mode.lower(), instrument, tag_object))

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
    print("Input tags: {}".format(ad.tags))
    print("Recipes available for the input file: ")

    for recipe in all_recipies:
        print("  " + recipe)
    return