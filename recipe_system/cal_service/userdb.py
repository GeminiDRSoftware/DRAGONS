# Defines the UserDB class for calibration returns
import os
import pickle

from .caldb import CalDB, CalReturn


class UserDB(CalDB):
    def __init__(self, name=None, user_cals=None, valid_caltypes=None,
                 log=None):
        super().__init__(name=name, store=True, log=log,
                         valid_caltypes=valid_caltypes)
        self.cachefile = os.path.join(self.caldir, "calindex.pkl")

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
            self.log.debug(f"Found {len(self.user_cache)} manual calibrations"
                           " in user cache file.")

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

    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
        # Return a list as long as adinputs if this calibration type is in
        # the user_cals
        if caltype in self.user_cals:
            return CalReturn([(self.user_cals[caltype], "user_cals")] *
                             len(adinputs))

        # Go through the list one-by-one, looking in the user_cache
        cals = []
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
            self.user_cache[ad.calibration_key, caltype] = calfile
        self.save_cache()

    def _unset_calibrations(self, adinputs, caltype=None):
        """Remove this calibration from the user_cache"""
        for ad in adinputs:
            del self.user_cache[ad.calibration_key, caltype]
        self.save_cache()

    def _clear_calibrations(self):
        self.user_cache = {}
        self.save_cache()
