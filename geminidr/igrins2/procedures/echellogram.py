import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class StripBase(object):
    def __init__(self, order, wvl, x, y, kind=None):
        self.order = order
        self.x = x
        self.y = y
        self.wvl = wvl

        if kind is None:
            if len(wvl) > 512:
                kind = "linear"
            else:
                kind = "cubic"

        self.interp_x = interp1d(self.wvl, self.x, kind=kind,
                                 bounds_error=False)
        self.interp_y = interp1d(self.wvl, self.y, kind=kind,
                                 bounds_error=False)

class Echellogram(object):
    def __init__(self, orders, wvl_x_y_list, kind=None):

        self.orders = orders
        self.zdata = {}
        for o, (wvl, x, y) in zip(orders, wvl_x_y_list):
            z = StripBase(o, wvl, x, y, kind=kind)
            self.zdata[o] = z

    @classmethod
    def from_json_fitted_echellogram_sky(cls, json_name):
        import json
        echel = json.load(open(json_name))

        wvl_x_y_list = []
        for wvl, y in zip(echel["wvl_sampled_list"],
                          echel["y_sampled_list"]):
            x = echel["x_sample"]
            wvl_x_y_list.append((wvl, x, y))

        obj = cls(echel["orders"], wvl_x_y_list)
        return obj

    @classmethod
    def from_aperture_and_wvlsol(cls, ap, wvlsol):

        assert ap.orders == wvlsol["orders"]

        n = len(wvlsol["wvl_sol"][0])

        xi = np.arange(n)
        yi_list = [ap(o, xi) for o in ap.orders]

        wvl_x_y_list = []
        for wvl, yi in zip(wvlsol["wvl_sol"],
                          yi_list):
            wvl_x_y_list.append((wvl, xi, yi))

        obj = cls(ap.orders, wvl_x_y_list, kind="linear")
        return obj


    def get_xy_list(self, lines_list):
        """
        get (x, y) list for given orders and wavelengths.

        lines_list : dict of wavelength list
        """
        zdata = self.zdata
        xy1 = []
        for order_i, wvl in lines_list.items():
            #print order_i
            if len(wvl) > 0 and order_i in zdata:
                zz = zdata[order_i]
                xy1.extend(zip(zz.interp_x(wvl), zz.interp_y(wvl)))

        return xy1

    def get_xy_from_wvl(self, o, wvl):
        zz = self.zdata[o]
        x, y = zz.interp_x(wvl), zz.interp_y(wvl)
        return x, y

    def get_xy_list_filtered(self, lines_list):

        xy_list = self.get_xy_list(lines_list)
        nan_filter = [np.all(np.isfinite([x, y])) for x, y in xy_list]
        xy1f = np.compress(nan_filter, xy_list, axis=0)

        return xy1f, nan_filter



    def save(self, fn):

        d = dict(orders=self.orders,
                 wvl_list=[],
                 x_list=[],
                 y_list=[])

        for o in self.orders:
            zd = self.zdata[o]
            d["wvl_list"].append(zd.wvl)
            d["x_list"].append(zd.x)
            d["y_list"].append(zd.y)

        from json_helper import json_dump
        json_dump(d, open(fn, "w"))

    @classmethod
    def load(cls, json_name):
        import json

        d = json.load(open(json_name))

        obj = cls.from_dict(d)

        return obj


    @classmethod
    def from_dict(cls, d):

        wvl_x_y_list = []
        for wvl, x, y in zip(d["wvl_list"],
                             d["x_list"],
                             d["y_list"]):
            wvl_x_y_list.append((wvl, x, y))

        obj = cls(d["orders"], wvl_x_y_list)

        return obj

    def get_df(self):
        df_echellogram = pd.concat([pd.DataFrame(dict(order=o, x=z.x, y=z.y, wvl=z.wvl))
                                    for o, z in self.zdata.items()])

        return df_echellogram


if __name__ == "__main__":
    echel_name = "fitted_echellogram_sky_H_20140316.json"
    echel = Echellogram.from_json_fitted_echellogram_sky(echel_name)
