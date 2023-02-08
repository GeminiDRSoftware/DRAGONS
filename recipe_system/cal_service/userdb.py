# Defines the UserDB class for calibration returns. The UserDB handles the
# following things:
#   - the user_cals provided to the Reduce() object
#   - any manually-assigned calibrations in the pickle
#   - standard MDFs

import os
import pickle

from .caldb import CalDB, CalReturn


class UserDB(CalDB):
    """
    The class handling the User-defined database, which can contain:
    (a) a filename for each caltype that should be returned whenever that
        caltype is requested, irrespective of the AD "making" the request
    (b) a filename to be returned for a given caltype for a specific AD
        (actually, a specific ad.calibration_key(), which is normally just
        the data label)

    This class also handles the standard MDFs when a "mask" is requested.

    Attributes
    ----------
    user_cals : dict
        A dict of {caltype: filename} as described in (a) above
    user_cache : dict
        A dict of {(calibration_key, caltype): filename} as described in (b)
        This is cached to disk as a pickle.
    mdf_dict : dict
        A dict of {mdf_key: filename} for the standard focal plane masks
    mdf_key : tuple
        A tuple of strings representing the attributes of the astrodata object
        to concatenate to form the keys of mdf_dict.
    """
    def __init__(self, name=None, get_cal=True, mdf_dict=None, mdf_key=None,
                 user_cals=None, valid_caltypes=None, log=None):
        super().__init__(name=name, get_cal=get_cal, store_cal=True,
                         log=log, valid_caltypes=valid_caltypes,
                         procmode=None)
        self.cachefile = os.path.join(self.caldir, "calindex.pkl")
        self.mdf_dict = mdf_dict
        self.mdf_key = mdf_key
        if self.mdf_dict:
            self.log.debug(f"Read {len(mdf_dict)} standard MDFs.")
            if not self.mdf_key:
                raise RuntimeError("No mdf_key supplied for mdf_dict.")

        self.user_cals = {}
        if user_cals:
            for caltype, calfile in user_cals.items():
                if caltype in self._valid_caltypes:
                    self.user_cals[caltype] = calfile
                    self.log.stdinfo(f"Manually assigned {calfile} as {caltype}")
                else:
                    raise ValueError("Unknown calibration type {!r}".format(caltype))

        self.user_cache = self.load_cache()
        if self.user_cache:
            self.log.stdinfo(f"Found {len(self.user_cache)} manual "
                             "calibrations in user cache file.")

    def load_cache(self):
        try:
            with open(self.cachefile, "rb") as fp:
                return pickle.load(fp)
        except OSError:
            self.log.debug("Cannot open user cache file.")
            return {}

    def save_cache(self):
        with open(self.cachefile, "wb") as fp:
            pickle.dump(self.user_cache, fp)

    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
                          howmany=1):
        # Return a list as long as adinputs if this calibration type is in
        # the user_cals. howmany is irrelevant, since there can be only one
        # match in the UserDB
        if caltype in self.user_cals:
            return CalReturn([(self.user_cals[caltype], "user_cals")] *
                             len(adinputs))

        cals = []
        if caltype == "mask" and self.mdf_dict is not None:
            for ad in adinputs:
                # The data necessary to distinguish which MDF to use is
                # different for different instruments, so mdf_key is defined
                # for each one and the key generated from it here.
                key = "_".join(getattr(ad, desc)() for desc in self.mdf_key)
                try:
                    mdf_file = self.mdf_dict[key]
                except KeyError:
                    cals.append(None)
                else:
                    cals.append((mdf_file, "standard masks"))
        else:
            # Go through the list one-by-one, looking in the user_cache
            for ad in adinputs:
                try:
                    cals.append((self.user_cache[ad.calibration_key(),
                                                 caltype], self.name))
                except KeyError:
                    cals.append(None)
        return CalReturn(cals)

    def _store_calibration(self, cal, caltype=None):
        # Method has no meaning for the manual overrides, "set" is used instead
        pass

    def _set_calibrations(self, adinputs, caltype=None, calfile=None):
        """Add this calibration to the user_cache"""
        for ad in adinputs:
            self.user_cache[ad.calibration_key(), caltype] = calfile
        self.save_cache()

    def _unset_calibrations(self, adinputs, caltype=None):
        """Remove this calibration from the user_cache"""
        for ad in adinputs:
            del self.user_cache[ad.calibration_key(), caltype]
        self.save_cache()

    def _clear_calibrations(self):
        self.user_cache = {}
        self.save_cache()
