from ..utils.list_utils import compress_list


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

    def save(self, fn):
        from .json_helper import json_dump
        json_dump(self.data, open(fn, "w"))

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
