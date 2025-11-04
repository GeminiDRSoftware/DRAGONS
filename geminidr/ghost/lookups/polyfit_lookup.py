import os
import glob

def get_polyfit_filename(log, arm, mode, date_obs, filename, caltype):
    """
    Gets the filename of the relevant initial polyfit file for this
    input GHOST science image by searching for files in directories of
    a known structure.

    This primitive uses the arm, resolution mode and observing epoch
    of the input AstroData object to determine the correct initial
    polyfit model to provide. The model provided matches the arm and
    resolution mode of the data, and is the most recent model generated
    before the observing epoch.

    Parameters
    ----------
    log: The recipe logger
    arm: The GHOST arm (blue, red, or slitv)
    mode: The GHOST mode (std or high)
    date_obs: ad.ut_date()
    filename: ad.filename
    caltype : str
        The initial model type (e.g. ``'rotmod'``, ``'spatmod'``, etc.)
        requested. An :any:`AttributeError` will be raised if the requested
        model type does not exist.

    Returns
    -------
    str/None:
        Filename (including path) of the required polyfit file
    """
    polyfit_dir = os.path.join(os.path.dirname(__file__),
                               'Polyfit')

    available = sorted(glob.glob(os.path.join(polyfit_dir, arm, mode, "*",
                                              f"{caltype}.fits")))
    if not available:
        if not log is None:
            log.warning(f"Cannot find any {caltype} files for {filename}")
        return None

    date_obs = date_obs.strftime("%y%m%d")
    calfile = None
    for av in available:
        if av.split(os.path.sep)[-2] <= date_obs:
            calfile = av

    if not calfile:
        if not log is None:
            log.warning(f"Cannot find a {caltype} for {filename} {date_obs}")
        return None

    return calfile
