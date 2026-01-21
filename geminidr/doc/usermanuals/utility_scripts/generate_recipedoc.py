#!/usr/bin/env python

"""
Generate .rst files to document the recipes.  The .rst files are
meant to be included in the sphinx doc.  The script scans geminidr and
find all the recipes and generate one rst for each.

For this to work, the section separators in the main sphinx document
must be followed exactly because the generated files use that convention.
"""

import sys
import argparse
import re

import os.path
import pkgutil
from importlib import import_module
import types
import inspect
import textwrap

SHORT_DESCRIPTION = "General documentation for recipes."

DOCUMENTED_INSTPKG = {'f2', 'gmos', 'gnirs', 'gsaoi', 'niri'}
DRPKG = 'geminidr'

def parse_args(command_line_args):
    """
    Parse the command line.
    """
    parser = argparse.ArgumentParser(description=SHORT_DESCRIPTION)
    parser.add_argument('pkgs', type=str, nargs="*",
                        help="List of packages to generate, eg. 'gmos'. "
                             "If none specified, every packages in 'geminidr'"
                             " will be generated.")
    parser.add_argument('--destination', '-d', nargs=1, dest='dest',
                        type=str, default='.', action='store', required=False,
                        help='Destination directory for the output files')
    parser.add_argument('--context', nargs=1, dest='context', type=str,
                        default='sq', action='store', required=False,
                        help='Which type of recipes? qa, ql, or sq')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='Toggle verbose mode')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Toggle debug mode')

    args = parser.parse_args(command_line_args)
    if isinstance(args.dest, str):
        args.dest = [args.dest]
    if isinstance(args.context, str):
        args.context = [args.context]

    if args.debug:
        print(args)

    return args

def generate_recipedoc(documented_instpkg, destination, context):
    """
    Primary function to generate the recipe rst documentation.
    """
    instpkgs = get_matching_instpkg(documented_instpkg, DRPKG)

    for instpkg in instpkgs:
        # get list of module (name). those are the recipe libraries.
        modulenames = get_list_of_recipe_modules(instpkg, DRPKG, context)

        # keyed on recipe name
        #   recipename_key = {'function': func, # the local one
        #                     'tagssets': [tagset1, tagset2, ..],
        #                     'default' : None/tagsets}

        recipes_to_document = {}

        for modulename in modulenames:
            #print(modulename)
            module = import_module(modulename)

            # Get the list of recipes (will need name, docstring, content)
            recipes_in_module = find_recipes(module)
            #print(recipes_in_module)
            for recipe in recipes_in_module:
                write_recipe_rst(recipe, modulename, destination)


                # print("recipe:", recipe)
                # print("module:",inspect.getmodule(recipe['function']))
                # try:
                #     print("tags:",module.recipe_tags)
                # except AttributeError:
                #     if 'recipes_common' in modulename:
                #         pass
                #     else:
                #         raise
                # print()
                # recipename = recipe['function'].__name__
                # mod_recipe_key = f'{modulename}.{recipename}'
                # if mod_recipe_key not in recipes_to_document:
                #     recipes_to_document[mod_recipe_key] = {'function': None,
                #                                            'tagsets': None,
                #                                            'default': False}
                # # if recipe['origin'] == 'local':
                # #     recipes_to_document[recipename]['function'] = recipe['function']
                # recipes_to_document[mod_recipe_key]['function'] = recipe['function']
                #
                # try:
                #     recipes_to_document[mod_recipe_key]['tagsets'] = module.recipe_tags
                # except AttributeError:
                #     if 'recipes_common' in modulename:
                #         pass
                #     else:
                #         raise
                #
                # if recipe['default']:
                #     recipes_to_document[mod_recipe_key]['default'] = True

        # for recipe in recipes_to_document:
        #     print("recipe name: ", recipes_to_document[recipe]['function'].__name__)
        #     print("module: ", inspect.getmodule(recipes_to_document[recipe]['function']))
        #     print("content of 'recipe' key:", recipes_to_document[recipe])
            #break

    return

def get_matching_instpkg(documented_instpkg, drpkg):
    """
    Walkthrough the drpkg and confirm that the requested instrument
    packages are in the approved list and are indeed packages.
    """
    drpkg_mod = import_module(drpkg)
    #drpkg_importer = pkgutil.ImpImporter(drpkg_mod.__path__[0])
    instpkgs = []
    #for pkgname, ispkg in drpkg_importer.iter_modules():
    for module_info in pkgutil.iter_modules(drpkg_mod.__path__):
        pkgname = module_info.name
        ispkg = module_info.ispkg
        if ispkg:
            if pkgname in documented_instpkg:
                instpkgs.append(pkgname)

    return instpkgs

def get_list_of_recipe_modules(instpkg, drpkg, context):
    """
    Find the names of all the modules in this instrument package.
    Keep only first level modules, it does not look in packages.
    """
    instpkg_module = import_module(os.extsep.join([drpkg, instpkg, 'recipes', context]))
    #instpkg_importer = pkgutil.ImpImporter(instpkg_module.__path__[0])
    modulenames = []
    #for modulename, ispkg in instpkg_importer.iter_modules():
    for module_info in pkgutil.iter_modules(instpkg_module.__path__):
        modulename = module_info.name
        ispkg = module_info.ispkg
        if ispkg:
            continue
        modulenames.append(os.extsep.join([instpkg_module.__name__, modulename]))

    return modulenames


def find_recipes(module):
    recipes_in_module = []

    all_names = []
    for name, func in inspect.getmembers(module, inspect.isfunction):

        try:
            tags = module.recipe_tags
        except AttributeError:
            if 'recipes_common' in module.__name__:
                tags = None
            else:
                raise

        if inspect.getmodule(func) == module:
            origin = 'local'
        else:
            origin = 'imported'

        entry = {'function': func,
                 'tags': tags,
                 'origin': origin,
                 'default': False}

        if name == '_default':
            entry['default'] = True

        if func.__name__ in all_names:
            i = all_names.index(func.__name__)
            if recipes_in_module[i]['default']:
                if entry['default']:  # 2 defaults, something is wrong
                    raise
                # else do skip this duplicate, keep the default
            else:
                if entry['default']:
                    dummy = all_names.pop(i)
                    dummy = recipes_in_module.pop(i)
                    all_names.append(func.__name__)
                    recipes_in_module.append(entry)
                else:  # same name, none default, something is wrong
                    raise
        else:
            all_names.append(func.__name__)
            recipes_in_module.append(entry)

    return recipes_in_module


def write_recipe_rst(recipe, modulename, destination):
    """
    Format the recipe information for importing into the rst recipe template.

    modulename is used for unique file name when a recipe is imported.
    Otherwise, the module one gets from inspecting the function is the parent
    module where the recipe really lives.
    """

    # Extract info from "recipe"

    # Name, modules, tags
    recipename = recipe['function'].__name__
    recipetags = recipe['tags']
    if recipe['origin'] == 'imported':
        recipe_library = f'{modulename}\n'
        recipe_imported = f'{inspect.getmodule(recipe["function"]).__name__}'
    else:
        recipe_library = modulename

    # Format the docstring to have the parameters in a code block.
    docstring = recipe['function'].__doc__
    if docstring:
        noindent_docstring = textwrap.dedent(docstring)
        param_regex = r'\b(Parameters\s+-*[\w\s().,;:-]+)'
        param_section = textwrap.indent(
            re.search(param_regex, noindent_docstring)[0],
            4 * " ")
        formatted_docstring = re.sub(param_regex, f'::\n\n{param_section}',
                                     noindent_docstring)
    else: # no docstring
        formatted_docstring = None
        print('WARNING: no docstring for ', modulename, recipename)

    # The source includes the docstring, we want just the recipe,
    # regex sub the docstring out.
    # regex from:
    # https://stackoverflow.com/questions/59270042/efficent-way-to-remove-docstring-with-regex
    #  TODO: regex for the ''' ''' style too.
    source_with_doc = inspect.getsource(recipe['function'])
    doc_reg = r'\b(def.+|class.+)\s+"{3}[\w\s()"\'`.,;:-]+"{3}'
    result = re.search(doc_reg, source_with_doc)
    source = re.sub(doc_reg, r'\1', source_with_doc)

    # define filename and open it.
    filename = f'{modulename}.{recipename}.rst'
    f = open(os.path.join(destination, filename), 'w')

    f.write(f'{recipename}\n')
    ndelim = len(recipename)
    f.write(f'{"="*ndelim}\n\n')

    f.write(f'| **Recipe Library**: {recipe_library}')
    if recipe['origin'] == 'imported':
        f.write(f'| **Recipe Imported From**: {recipe_imported}')
    f.write(f'\n| **Astrodata Tags**: {recipetags}\n')

    if formatted_docstring:
        f.write(f'{formatted_docstring}\n')

    #f.write(f'**Recipe**::\n\n')
    f.write(f'::\n\n')
    f.write(f'{textwrap.indent(source, 4*" ")}\n')

    f.close()




    return

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.pkgs:
        documented_instpkg = set(args.pkgs)
    else:
        documented_instpkg = DOCUMENTED_INSTPKG

    if not os.path.isdir(args.dest[0]):
        os.makedirs(args.dest[0])

    generate_recipedoc(documented_instpkg, args.dest[0], args.context[0])

if __name__ == '__main__':
    sys.exit(main())


