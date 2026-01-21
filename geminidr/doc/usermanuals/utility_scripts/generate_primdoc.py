#!/usr/bin/env python

"""
Generate .rst files to be included in the sphinx doc.  This is meant to
avoid duplication and having to maintain two locations for the information.
The script will scan geminidr and find all the various implementation of the
primitives and note all the different defaults, and create formatted files
that can then be ".. include"d into the primitive main .rst file.

For this to work, the separators in the main sphinx document must be
followed exactly because the generated files use that convention.

Usage:
  generate_primdoc -d primitives/generated-doc
"""

import sys
import argparse

import os.path
import pkgutil
from importlib import import_module
import types
import inspect
import textwrap

from gempy.library import config

SHORT_DESCRIPTION = "Generate primitive doc from docstrings and pex.config class definitions."

DOCUMENTED_INSTPKG = {'core', 'f2', 'gemini', 'gmos', 'gnirs', 'gsaoi', 'niri'}
DRPKG = 'geminidr'

PARAMHEADER = 'Parameter defaults and options\n' \
              '------------------------------\n' \
              '::\n\n'

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
                        help='destination directory for the output files')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='Toggle verbose mode')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Toggle debug mode')

    args = parser.parse_args(command_line_args)

    if args.debug:
        print(args)

    return args

def generate_primdoc(documented_instpkg, destination):
    """
    primary function to generate the primitive rst documentation.
    """
    instpkgs = get_matching_instpkg(documented_instpkg, DRPKG)

    for instpkg in instpkgs:
        # get list of module (name). those are the primitives_ and
        # parameters_ modules for the most part.
        modulenames = get_list_of_modules(instpkg, DRPKG)

        for modulename in modulenames:
            module = import_module(modulename)

            # find primitives and parameter config classes in each module.
            primclasses_in_module, paramclasses_in_module = \
                    find_prims_params(module)

            # generate doc for all primitives in primitive classes
            if primclasses_in_module:
                for primclass in primclasses_in_module:
                    write_primitives_rst(primclass, destination)

            if paramclasses_in_module:
                for paramclass in paramclasses_in_module:
                    write_parameters_rst(paramclass, destination,
                                         module)

    return

def get_matching_instpkg(documented_instpkg, drpkg):
    """
    walkthrough the drpkg and confirm that the requested instrument
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

def get_list_of_modules(instpkg, drpkg):
    """
    Find the names of all the modules in this instrument package.
    Keep only first level modules, it does not look in packages.
    """

    instpkg_module = import_module(os.extsep.join([drpkg, instpkg]))
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

def find_prims_params(module):
    """
    Go through a module and find the all primitive classes  and/or
    all the parameter classes within.

    In our current geminidr implementation, there is generally only one
    primitive class per module.  Also, we generally do not mix primitive
    classes and parameter classes within the same module, we keep them separate.

    Normally, one of primclasses_in_module or paramclasses_in_module will be
    an empty list.  But that can change, this function can deal with that.
    """
    primclasses_in_module = []
    paramclasses_in_module = []

    for name, clss in inspect.getmembers(module, inspect.isclass):
        # exceptions to skip
        if clss.__module__ == 'geminidr.core.parameters_generic':
            continue
            # This parameter module is not associated with any primitive
            # module.  Just skip it.
        if name == 'core_stacking_config':
            continue
            # This is a utility parameter config class. Skip it.

        if clss.__module__ == module.__name__ and hasattr(clss, 'tagset'):
            primclasses_in_module.append(clss)
        elif issubclass(clss, config.Config):
            paramclasses_in_module.append(clss)

    return primclasses_in_module, paramclasses_in_module

def write_primitives_rst(primclass, destination):
    """
    Get the docstring and formatted pex.config parameters to rst files, one
    for the docstring, one for the pex.config stuff (to catch the active
    defaults).

    """
    try:
        primclass_instance = primclass(None)
    except ValueError:
        # GMOSLongslit is a special "magic" class that selects either the
        # GMOSClassicLongslit or the GMOSNSLongslit.  I cannot be instanciated
        # with None.
        if primclass.__name__ == 'GMOSLongslit':
            return

    for name, item in primclass.__dict__.items():
        if name.startswith('_'):
            continue
        if isinstance(item, types.FunctionType):
            rootfilename = f'{primclass.__module__}.{primclass.__name__}.{name}'

            # first the docstring
            filename = f'{rootfilename}_docstring.rst'
            f = open(os.path.join(destination, filename), 'w')
            docstring = getattr(primclass_instance, name).__doc__
            if docstring:
                f.write(textwrap.dedent(docstring))
            else: # no docstring
                f.write('')
                print('WARNING: no docstring for ', primclass.__name__, name)
            f.close()

            # then the overridden parameters
            params = primclass_instance.params[name]
            filename = f'{rootfilename}_param.rst'
            f = open(os.path.join(destination, filename), 'w')
            f.write(PARAMHEADER)
            for k, v in params.items():
                if not k.startswith("debug"):
                    formatted_doc = params.doc(k).replace('\n', '\n      ')
                    f.write(f'   {k:20s} {v!r:20s} {formatted_doc}\n')
            f.close()
    return

def write_parameters_rst(paramclass, destination, module):
    """
    Get the pex.config parameter list and defaults and format for sphinx.
    This function is highly customize to the way we implement things in
    DRAGONS.  It does assume that a given "parameter_blah" module is meant
    to match the "primitive_blah" module.  We need to access the parameters
    though the primitive to insure the correct inheritance is followed.

    (See commented cryptic note at the bottom of this file.)
    """
    associated_primodulename = module.__name__.replace('parameters',
                                                       'primitives')
    associated_primname = paramclass.__name__.split('Config')[0]

    print("KLDEBUG: associated_primodulename=", associated_primodulename)
    print("KLDEBUG: associated_primname=", associated_primname)

    primmod = import_module(associated_primodulename)
    for name, clss in inspect.getmembers(primmod, inspect.isclass):
        if clss.__module__ == primmod.__name__ and hasattr(clss, 'tagset'):
            try:
                clss_instance = clss(None)
            except ValueError:
                # GMOSLongslit is a special "magic" class that selects either the
                # GMOSClassicLongslit or the GMOSNSLongslit.  I cannot be instanciated
                # with None.
                if clss.__name__ == 'GMOSLongslit':
                    continue


            params = clss_instance.params[associated_primname]

            rootfilename = f'{primmod.__name__}.{clss.__name__}.{associated_primname}'

            filename = f'{rootfilename}_param.rst'
            f = open(os.path.join(destination, filename), 'w')
            f.write(PARAMHEADER)
            for k, v in params.items():
                if not k.startswith("debug"):
                    formatted_doc = params.doc(k).replace('\n', '\n      ')
                    f.write(f'   {k:20s} {v!r:20s} {formatted_doc}\n')
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

    generate_primdoc(documented_instpkg, args.dest[0])

if __name__ == '__main__':
    sys.exit(main())




## KL Development Notes
# Parameters classes.  I will need to do module name parsing to associate with
# the primitive.
# I see two ways:
#    1) from the prim and assumes that "primitives" can be replaced with "parameters".
#       then check if the param module exists, then if the prim param are overridden.
#       Remember that a prim default can be in the param module without being
#       in the prim module.
#       This technique would be more reliable if an attribute was saved
#       with the full import path to overriding parameter file being used.
#       Without that though, it seems no more risky than #2 and more
#       complicated.
#    2) from the param, when it exists, then asssume that "parameters" can be
#       replaced with "primitives".  Then assume name of prim is the bit pre-"Config".
#       This is what is currently implemented.
#

#    For the parameters doc, I need to add only a parameter section to the doc when
#    only the parameters or defaults are changed.  Eg. if everything else is as
#    the generic, just different defaults, no point in repeating the docstring.

# make testdoc
# utility_scripts/generate_primdoc.py core gmos -d testdoc