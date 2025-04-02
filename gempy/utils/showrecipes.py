#!/usr/bin/env python
import os
import sys
import re
import inspect
import importlib

import astrodata
import gemini_instruments
import geminidr

from astrodata import AstroDataError
from recipe_system.utils.errors import ModeError
from recipe_system.utils.errors import RecipeNotFound
from recipe_system.mappers.recipeMapper import RecipeMapper


def showrecipes(_file, adpkg=None, drpkg=None):
    """
    Takes in a file, and will return all the possible recipes that can be used
    on the file.

    Parameters
    ----------
    _file : str
        file name for the fits file

    Returns
    -------
    str
        String with all the information provided
    """
    # string that results are appended to
    result = ""

    if adpkg is not None:
        importlib.import_module(adpkg)

    # Find the file and open it with astrodata
    try:
        ad = astrodata.from_file(_file)
        tags = ad.tags
    except AstroDataError:
        result += ("There was an issue using the selected file, please check "
                   "the format and directory.")
        raise

    # New Mapper interface now requires a tag set and instrument name.
    # We decoupled the tags from 'ad'.
    dtags = set(list(ad.tags)[:])
    instpkg = ad.instrument(generic=True).lower()

    all_recipies = []
    functions_list = []
    for mode in ['sq', 'qa', 'ql']:

        try:
            if drpkg is None:
                rm = RecipeMapper(dtags, instpkg, mode=mode)
            else:
                rm = RecipeMapper(dtags, instpkg, mode, drpkg=drpkg)

            recipe = rm.get_applicable_recipe()

        except RecipeNotFound:
            error_message = "The RecipeMapper returned a RecipeNotFound " \
                            "error. For showrecipes, this means that " \
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
                              inspect.isfunction(i[1]) and (i[0] != '_default')]

            # Appends said recipes to an external list so it can be called later
            for i in range(len(functions_list)):
                all_recipies.append(recipe.__module__ + "::"
                                    + functions_list[i])

    result += ("Input file: {}".format(os.path.abspath(ad.path)))

    result += ("\nInput tags: {}".format(tags))

    # Edge case exists where ql mode isn't implemented, sq/qa both pass due to
    # except clause, and then no recipes were found.

    if not all_recipies:
        result += "\n!!! No recipes were found for this file !!!"
    else:
        result += "\nRecipes available for the input file: "
        for recipe in all_recipies:
            result += "\n   " + recipe

    return result


def showprims(_file, mode='sq', recipe='_default', adpkg=None, drpkg=None):
    """
    showprims takes in a file, observing mode, and the data reduction
    recipe name, and will return the source code pertaining to that recipe,
    straight from the source file. This source codes shows all the primitives
    that are used, helping user distinguish what primitives are used.

    Parameters
    ----------
    _file : str
        file name for the fits file, useful for the astrodata.tags, which are
        then used to determine what directory to find the recipe in.
    mode : str
        either 'qa' or 'sq' for the two modes available. Anything else will
        return an error. mode also needed to navigate to the proper directory
    recipe : str
        name of the recipe. Must be exactly the same as the function
        name/recipe. str + mode find the python file where the recipe exists
        in, the parameter 'recipe' is needed to isolate the exact recipe (if
        there are multiple) that the user wants to see the source code from.

    Returns
    -------
    str
        String with all the information provided.
    """
    # string that results are appended to
    result = ""

    if adpkg is not None:
        importlib.import_module(adpkg)

    # Make sure mode is a valid input
    if mode.lower() not in ['sq', 'qa', 'ql']:
        raise ValueError("mode must be 'sq', 'qa', or 'ql'!")

    # Find the file and open it with astrodata
    try:
        ad = astrodata.from_file(_file)
        tags = ad.tags
    except AstroDataError:
        print("There was an issue using the selected file, please check "
              "the format and directory:", sys.exc_info()[0])
        raise

    # New Mapper interface now requires a tag set and instrument name.
    # We decoupled the tags from 'ad'.
    dtags = set(list(ad.tags)[:])
    instpkg = ad.instrument(generic=True).lower()
    try:
        if drpkg is None:
            rm = RecipeMapper(dtags, instpkg, mode=mode, recipename=recipe)
        else:
            rm = RecipeMapper(dtags, instpkg, mode=mode, recipename=recipe,
                              drpkg=drpkg)
        mapper_recipe = rm.get_applicable_recipe()

    except RecipeNotFound:
        if recipe == '_default':

            error_message = \
                "Recipe was not provided, and the module that corresponded " \
                "to the file (and mode) provided did not specify a default " \
                "recipe to use. Please make sure a recipe is provided. \n" \
                "\n Mode = {}  \n Recipe = {}".format(mode, recipe)

        else:
            error_message = \
                "The RecipeMapper returned a RecipeNotFound error. For " \
                "showprims, this may mean that there does not exist " \
                "a recipe for the given file. This may be because the tags" \
                "given as input do not match any recipe tags"

        raise RecipeNotFound(error_message)

    except ModeError:
        raise ModeError("The mode provided either does not exist, or has not "
                        "been implemented yet. Because of this, no recipe "
                        "can be found for it.")

    else:
        mod = importlib.import_module(mapper_recipe.__module__)

        # Retrieves all the source code of the function
        source_code = inspect.getsource(mapper_recipe)

        # Just helps the user understand that the default recipe was used
        if recipe == '_default':
            result += ("Recipe not provided, default recipe ({}) will "
                       "be used.\n".format(mapper_recipe.__name__))

    result += ("Input file: " + os.path.abspath(ad.path))
    result += ("\nInput tags: {}".format(list(tags)))
    result += ("\nInput mode: " + str(mode.lower()))
    result += ("\nInput recipe: " + mapper_recipe.__name__)
    result += ("\nMatched recipe: " + mapper_recipe.__module__ + "::" +
               mapper_recipe.__name__)
    result += ("\nRecipe location: " + mapper_recipe.__globals__['__file__'])
    result += ("\nRecipe tags: " + str(mod.recipe_tags))
    result += "\nPrimitives used: "

    for primitive in re.findall(r'p\..*', source_code):
        result += ("\n   " + primitive)

    return result
