import inspect
import astrodata
import gemini_instruments
import re
import importlib
import sys
import os


def show_primitives(_file, mode='sq', recipe='reduce'):
    """
    show_primitives takes in a file, observing mode, and the data reduction
    recipe name, and will return the source code pertaining to that recipe,
    straight from the source file. This source codes shows all the primitives
    that are used, helping user distinguish what primitives are used.

    Parameters
    ----------
    file - str
        file name for the fits file, useful for the astrodata.tags, which are
        then used to determine what directory to find the recipe in.
    mode - str
        either 'qa' or 'sq' for the two modes available. Anything else will
        return an error. mode also needed to navigate to the proper directory
    recipe - str
        name of the recipe. Must be exactly the same as the function
        name/recipe. str + mode find the python file where the recipe exists
        in, the parameter 'recipe' is needed to isolate the exact recipe (if
        there are multiple) that the user wants to see the source code from.

    Returns - None
    Prints out information
    -------
    """

    # Make sure mode is a valid input
    if mode.lower() not in ["sq", "qa"]:
        raise ValueError("mode must be 'sq' or 'qa'!")

    # Find the file and open it with astrodata
    try:
        if os.path.isabs(_file):
            ad = astrodata.open(_file)
            assert ad.tags
        elif not os.path.isabs(_file):
            ad = astrodata.open(os.path.join(os.getcwd(), _file))
        else:
            raise OSError("Could not find file. The file provided was neither"
                          " an absolute file location or part of the current "
                          "\n working directory. Please check if file exists "
                          " in path provided")
    except:
        print("There was an issue using the selected file, please check"
              "the format and directory:", sys.exc_info()[0])
        raise

    # a list of current instruments needs to be found so show_primitives can
    # parse through those directories to find the recipes. Can't assume
    # static list, user may add instrument
    list_of_found_instruments = []

    # returns every folder/subfolders (which need to be parsed out)
    all_folders = [x[0] for x in os.walk(
        os.path.expanduser("~/workspace/dragons/geminidr/"))]

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
        " recipies exist in this directory, and no folder was" \
        " associated with the name of the instrument provided.\n" \
        " Check to see if the file provided has an instrument" \
        " associated with it, and that the instrument exists in /geminidr"

    instrument = ad.instrument().lower()

    if instrument in ["gmos-s", "gmos-n"]:
        instrument = "gmos"

    assert instrument in list_of_found_instruments, intersect_string

    # Finds of the file is DARK, FLAT, BIAS, NS or IMAGE so import_module
    # can import the correct module to obtain the proper recipe
    # TODO: this is hardcoded and does not consider new types of modules.
    # If that needs to be added, it is possible.
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

    # sets up full path to import, dynamically finds all characteristics
    absolute_dir = 'geminidr.' + instrument + '.recipes.' + \
                   mode.lower() + '.' + module

    # Tests to make sure the discovered path where the recipe is stores exists
    absolute_path = absolute_dir.replace(".", "/")
    exp_usr = os.path.expanduser("~/workspace/dragons/" +
                                 absolute_path + ".py")
    try:
        assert (os.path.exists(exp_usr))
    except AssertionError:
        raise ModuleNotFoundError(
            "The expected location of the recipe does not exist. "
            "'show_primitives' found the instrument to be '{}', the mode \n"
            "to be '{}', and the object to be '{}', but could not find a"
            " module in the expected directory: '{}.recipes.{}.{}.py'. \n"
            "This may mean the object type in the file provided does not "
            "have a recipe with the given mode and recipename "
            "parameters"
                .format(instrument, mode.lower(), tag_object,
                        instrument, mode.lower(), module))

    # creates the import statement, BUT DOES NOT import directory yet.
    mod = importlib.import_module(absolute_dir)

    # Used show_recipes to give user all possible recipes
    # if the recipe provided does not exist
    functions_list = [i[0] for i in inspect.getmembers(mod) if
                      inspect.isfunction(i[1])]

    # If recipe not provided/'default' input used, check if
    # default recipe exists, otherwise error
    if recipe is 'default':
        if 'default' in functions_list:
            print("Recipe was not provided, but file contained default "
                  "recipe. Show_primitives will use the default recipe.")
            recipe = 'default'
        else:
            raise ValueError(
                "Recipe was not provided, and the module that corresponded "
                "to the file and mode provided did not specify a default "
                "recipe to use. Please make sure a recipe is provided. \n \n"
                "The recipes found for this file and mode combination "
                "are: {}".format(functions_list))

    # Makes sure recipe exists in the module imported above.  TIP: This also
    #  protects any unwanted input from recipe, as it is used in an eval()
    # later, eval() can be unsafe to use without checking user input

    try:
        assert (recipe in dir(mod))
    except AssertionError:
        # Failure in assertion error means file does not exist in module.
        raise ImportError(
            "\n The requested recipe was not found in the expected module, "
            "more likely than not, this is because the recipe name given "
            "(in this case:'{}'), has a typo or isn't a real recipe. "
            "Please check parameters. \n \n"
            "The recipes found for this file and mode combination "
            "are: {}".format(recipe, functions_list))

    # show_primitives only imports the one function(recipe) that was requested.
    # eval is used to append strings and create an object for getsource.
    recipe_in_module = eval("mod." + recipe)

    # Retrieves all the source code of the function
    source_code = inspect.getsource(recipe_in_module)

    # output
    print("Input file: .{}".format(_file))
    print("Input mode: {}".format(mode.lower()))
    print("Input recipe: {}".format(recipe_in_module.__name__))
    print("Matched recipe: {}".format(absolute_dir + "::" +
                                      recipe_in_module.__name__))
    print("Primitives used:")

    for primitive in re.findall(r'p\..*', source_code):
        print("  " + primitive)
    return
