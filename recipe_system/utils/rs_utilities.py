"""
Set of functions in support of the recipe_system.

"""
import errno
import itertools

from os import makedirs
from os.path import join


def makedrpkg(pkgname, instruments, modes=None):
    """
    Create the basic structure for a data reduction package that the
    recipe system will recognize.

    Parameters
    ----------
    pkgname: str
        Name of the new dr-package.

    instruments: list of str
        Name of the instrument(s) for which to create a directory structure.

    modes: list of str
        Name of the recipe modes that will be supported.
        Eg. modes = ['sq', 'qa']   Default: ['sq']
        ('sq'=science quality, 'qa'=quality assessement)

    """

    def touch(fname):
        with open(fname, 'a'):
            return

    if modes is None:
        modes = ['sq']

    for (instr, mode) in itertools.product(instruments, modes):
        modpath = join(pkgname, instr, 'recipes', mode)
        try:
            makedirs(modpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
        # when ready to ditch python 2.7, replace try except with this:
        # makedirs(modpath, exist_ok=True)
        touch(join(modpath, '__init__.py'))

    for instr in instruments:
        touch(join(pkgname, '__init__.py'))
        touch(join(pkgname, instr, '__init__.py'))
        touch(join(pkgname, instr, 'recipes', '__init__.py'))

    return
