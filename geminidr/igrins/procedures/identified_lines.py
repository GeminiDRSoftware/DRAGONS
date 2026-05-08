import warnings
import numpy as np
import pandas as pd

from .line_identify_simple import match_lines1_pix


def compress_list(mask, items):
    return [o for (m, o) in zip(mask, items) if m]


class IdentifiedLines(object):
    def __init__(self, json=None):
        self.data = dict(wvl_list=[], ref_indices_list=[],
                         pixpos_list=[], orders=[])
        if json is not None:
            self.update(json)

    @classmethod
    def load(cls, j):
        k = cls()
        # import json
        # j = json.load(open(fn))
        k.update(j)

        return k

    # def save(self, fn):
    #     """
    #     fn: output file name or an file-like object.
    #     """
    #     from .json_helper import json_dump
    #     json_dump(self.data, open(fn, "w"))

    def dumps(self):
        from .json_helper import json_dumps
        return json_dumps(self.data)

    def update(self, d):
        self.data.update(d)

    def append_order_info(self, order, wvl, indices, pixpos):

        self.data["orders"].append(order)
        self.data["wvl_list"].append(wvl)
        self.data["ref_indices_list"].append(indices)
        self.data["pixpos_list"].append(pixpos)

    def get_dict(self):
        l = self.data
        ref_map = dict(zip(l["orders"],
                           zip(l["pixpos_list"],
                               l["ref_indices_list"],
                               l["wvl_list"])))
        return ref_map

    def _get_msk_list(self):
        pixpos_list = self.data["pixpos_list"]
        msk_list = [[(p >= 0) for p in pl] for pl in pixpos_list]

        return msk_list

    def get_xy_list_from_pixlist(self, ap):

        pixpos_list = self.data["pixpos_list"]
        msk_list = self._get_msk_list()

        pixpos_list2 = [compress_list(msk, pl) for (msk, pl)
                        in zip(msk_list, pixpos_list)]

        xy_list = ap.get_xy_list(dict(zip(self.data["orders"],
                                          pixpos_list2)))

        return xy_list

    def get_xy_list_from_wvllist(self, echellogram):

        msk_list = self._get_msk_list()

        wvl_list = self.data["wvl_list"]
        wvl_list2 = [compress_list(msk, wl) for (msk, wl) in zip(msk_list,
                                                                 wvl_list)]

        wvl_dict = dict(zip(self.data["orders"], wvl_list2))
        xy_list = echellogram.get_xy_list(wvl_dict)

        return xy_list

    def get_df(self):
        dfl = []
        for o, pixpos, wvl in zip(self.data["orders"],
                                  self.data["pixpos_list"],
                                  self.data["wvl_list"]):
            df1 = pd.DataFrame(dict(order=o, xpos=pixpos, wvl=wvl))
            dfl.append(df1)

        df = pd.concat(dfl).reset_index()

        return df

    def reidentify_specs(self,
                         orders, specs, tr_ref_to_tgt,
                         offset_threshold=1):
        ref_map = self.get_dict()
        # a dictionary of [pixel_positions_of_lines, their_index_in_the_reference,
        # known_wavelength] by their orders.

        identified_lines_tgt = IdentifiedLines()
        # identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
        #                                  pixpos_list=[], orders=[]))

        # For each target spsectrum, we try to locate lines. The known position of
        # lines in the reference spectrum is transformed to the target spectra
        # space using the tr_ref_to_tgt.
        for o, s in zip(orders, specs):
            if (o not in ref_map) or (o not in tr_ref_to_tgt):
                wvl, indices, pixpos = [], [], []
            else:
                pixpos, indices, wvl = ref_map[o]
                pixpos = np.array(pixpos)
                msk = (pixpos >= 0)

                ref_pix_list = tr_ref_to_tgt[o](pixpos[msk])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'Degrees of freedom')
                    pix_list, dist = match_lines1_pix(s, ref_pix_list)

                pix_list[dist > offset_threshold] = -1
                # FIXME Is it okay to use the 1 pixel threshold?
                pixpos[msk] = pix_list

            identified_lines_tgt.append_order_info(int(o), wvl, indices, pixpos)

        return identified_lines_tgt
