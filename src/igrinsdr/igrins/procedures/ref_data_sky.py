import numpy as np

from . import ohline_grouped


class OHLines(object):
    def __init__(self, fn):
        ohline_ = np.genfromtxt(fn)
        self.um = ohline_[:,0]/1.e4
        self.intensity = ohline_[:,1]/10.
        self._update_wavelengths()

    def _update_wavelengths(self):
        for lines in ohline_grouped.line_groups:
            for l in lines:
                i, wvl = l
                self.um[i] = wvl


def load_sky_ref_data(ref_loader):

    ref_ohline_indices_map = ref_loader.load("OHLINES_INDICES_JSON")

    ref_ohline_indices = ref_ohline_indices_map[ref_loader.band]

    ref_ohline_indices = dict((int(k), v) for k, v \
                              in ref_ohline_indices.items())

    fn = ref_loader.query("OHLINES_JSON")
    ohlines = OHLines(fn)

    r = dict(#ref_date=ref_utdate,
             # band=band,
             ohlines_db = ohlines,
             ohline_indices=ref_ohline_indices,
             )

    return r
