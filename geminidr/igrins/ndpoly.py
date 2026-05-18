######
# Helper function and classes for volume fit

from collections import namedtuple
from itertools import product

import numpy as np


class NdPoly:
    """This class is to help fitting n-dim data with 3-dimensional polynomial.
    Asume that we 3 independent variable of x, y, z, with polynomial order of
    Ox, Oy and Oz, then there will be (Ox+1)(Oy+1)(Oz+1) coeeficients. For example,
    Ox, Oy and Oz of (2, 2, 1), then v = c1*x^2*y^2*z + c2*x^2*y^2 + ... + c18.
    The `get_array` method will return [x^2*y^2*z, x^2*y^2, ...., 1] so that this
    can be used with least square method to get the coefficients of [c1, c2, ..., c18].
    """
    def _setup(self, orders, orderT, names):
        po_list = [orderT(*_) for _ in product(*list(range(o + 1) for o in orders))]

        self.orders = orderT(*orders)
        self.names = names
        self.orderT = orderT
        self.po_list = po_list

    def __init__(self, orders):
        names = range(len(orders))
        orderT = tuple

        self._setup(orders, orderT, names)

    def multiply(self, vv, coeffs):
        v = 0.
        for po, p in zip(self.po_list, coeffs):
            pod = po._asdict()
            v1 = np.multiply.reduce([pow(vv[k], pod[k])
                                     for k in self.names])
            v += p*v1

        return v

    def get_array(self, vv):
        v_list = []
        for po in self.po_list:
            pod = po._asdict()
            v1 = np.multiply.reduce([pow(vv[k], pod[k])
                                     for k in self.names])
            v_list.append(v1)

        return v_list

    def _get_frozen_p(self, k_survived):
        p = NdPoly([self.orders[_k] for _k in k_survived])
        return p

    def freeze(self, k, v, coeffs):

        k_survived = tuple(_k for _k in self.names if _k != k)
        p = self._get_frozen_p(k_survived)
        # p = NdPoly([self.orders[_k] for _k in k_survived], k_survived)

        poo = dict((_k, []) for _k in p.po_list)

        for c1, po in zip(coeffs, self.po_list):
            _ = (o for _k, o in zip(self.names, po) if _k != k)
            nk = p.orderT(_)
            oo = po[k]
            poo[nk].append(c1 * pow(v, oo))

        sol1 = [np.sum(poo[po]) for po in p.po_list]

        return p, sol1

    def to_pandas(self, **kwargs):
        """
        convert to pandas dataframe.
        """

        import pandas as pd
        index = pd.MultiIndex.from_tuples(self.po_list, names=self.names)
        df = pd.DataFrame(index=index, data=kwargs)

        return df

    @staticmethod
    def from_pandas(df):
        # df.index.values
        # df.index.names
        # df = coeffs

        orders = df.index.values.max(axis=0)
        p = NdPolyNamed(orders, df.index.names)

        coeffs = df.loc[p.po_list].values.reshape([-1,])

        return p, coeffs

        # coeffs = pd.read_json("coeffs.json", orient="split")


class NdPolyNamed(NdPoly):
    def __init__(self, orders, names):
        # orderT = my_namedtuple("order_" + "_".join(names), names)
        orderT = namedtuple("order_" + "_".join(names), names)

        self._setup(orders, orderT, names)

    def _get_frozen_p(self, k_survived):
        p = NdPolyNamed([self.orders[_k] for _k in k_survived], k_survived)
        return p

