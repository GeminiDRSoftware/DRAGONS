#!/usr/bin/env python
# -*- coding: utf8 -*-=
import os
import sys
import inspect
import importlib

import astrodata
import gemini_instruments
from astrodata.core import AstroDataError
from recipe_system.utils.errors import ModeError
from recipe_system.utils.errors import RecipeNotFound
from recipe_system.mappers.recipeMapper import RecipeMapper


def show_recipes(_file):
    """
    show_recipes takes in a file, and will return all the possible recipes
    that can be used on the file.

    Parameters
    ----------
    file - str
        file name for the fits file

    Returns - str
        returns  string with all the information provided
    -------
    """
    # string that results are appended to
    result = ""

    # Find the file and open it with astrodata
    try:
        ad = astrodata.open(_file)
        tags = ad.tags
    except AstroDataError:
        result  += ("There was an issue using the selected file, please check"
                    "the format and directory:", sys.exc_info()[0])
        raise

    all_recipies = []
    functions_list = []
    for mode in ['sq', 'qa', 'ql']:

        try:

            rm = RecipeMapper([ad], mode)
            recipe = rm.get_applicable_recipe()

        except RecipeNotFound:
            error_message = "The RecipeMapper returned a RecipeNotFound " \
                            "error. For show_recipes, this means that " \
                            "there does not exist a recipe for the given " \
                            "file. This may be because the observation type " \
                            "found in the astrodata tags does not match any" \
                            "module for the given mode"

            # If on the last run of the for loop, function_list is still empty,
            # and a RecipeNotFound error was raised again, then raise error.
            # Else module may just not exist for the current mode,
            if functions_list == [] and mode == mode[-1]:
                raise RecipeNotFound(error_message)

            else:
                pass

        except ModeError:
            # ModeError returned if the mode does not exist for the file
            # ql is not implemented yet, so this catches that exception
            pass

        else:
            mod = importlib.import_module(recipe.__module__)

            # finds all functions(recipies) in the module except for 'default'
            functions_list = [i[0] for i in inspect.getmembers(mod) if
                              inspect.isfunction(i[1]) and (i[0] != 'default')]

            # Appends said recipes to an external list so it can be called later
            for i in range(len(functions_list)):
                all_recipies.append(recipe.__module__ + "::"
                                    + functions_list[i])

    # Todo: ad.path may be updated to always return absolute path, if that
    # happens, remove os.getcwd()
    result += ("Input file: " + format(os.path.normpath(
        os.path.join(os.getcwd() + "/" + ad.path))))


    result += ("\nInput tags: {}".format(tags))

    # Edge case exists where ql mode isn't implemented, sq/qa both pass due to
    # except clause, and then no recipes were found.

    if all_recipies == []:
        result += ("\n!!! No recipes were found for this file !!!")
    else:
        result += ("\nRecipes available for the input file: ")
        for recipe in all_recipies:
            result += ("\n   " + recipe)

    return result

