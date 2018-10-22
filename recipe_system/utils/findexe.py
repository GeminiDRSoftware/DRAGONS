#
#                                                                  gemini_python
#
#                                                    recipe_system.utils.findexe
# ------------------------------------------------------------------------------
import os
import psutil


def findexe(exe):
    """
    Function receives an executable name, 'exe', as a string, and returns
    a list of pids that match the name.

    Parameters
    ----------
    exe : str
        Name of a running executable.

    Returns
    -------
    pids: list
        a list of extant pids found running 'exe'

    Example
    -------

    >>> findexe('autoredux')
    [98684]
    >>> findexe('emacs')
    [41273, 55557]

    """
    pids = []
    for proc in psutil.process_iter():
        pinfo = proc.as_dict(attrs=['pid', 'cmdline'])
        cl = pinfo['cmdline']
        if not cl:
            continue
        else:
            arg1 = cl[0]
        try:
            arg2 = cl[1]
        except IndexError:
            arg2 = ''

        subarg1 = os.path.split(arg1)[1].strip()
        subarg2 = os.path.split(arg2)[1].strip() if arg2 else arg2
        if exe == subarg1 or exe == subarg2:
            pids.append(pinfo['pid'])

    return pids
