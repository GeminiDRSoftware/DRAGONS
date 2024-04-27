from collections import namedtuple


_ShiftedImages = namedtuple("ShiftedImages",
                            ["image", "variance", "profile_map", "mask"])


class ShiftedImages(_ShiftedImages):
    def to_hdul(self):
        hdu_list = []
        for k in self._fields:
            # FIXME : replace EXTNAME with proper name
            hdu_list.append(([("EXTNAME", k.upper())], getattr(self, k)))
        return hdu_list

    @classmethod
    def from_hdul(cls, hdul):
        _ = [hdu.data.astype(bool) if hdu.header["EXTNAME"] == "MASK"
             else hdu.data
             for hdu in hdul]
        return cls(*_)
