import numpy as np

from itertools import product
from collections import namedtuple


def my_namedtuple(n, names):

    _kls = namedtuple(n, names)

    class kls(_kls):
        def __new__(cls, args):
            return super(kls, cls).__new__(cls, *args)

        def __getitem__(self, k):
            if k in self._fields:
                return getattr(self, k)
            else:
                return super(kls, self).__getitem__(k)

    kls.__name__ = n

    return kls


class NdPoly(object):
    def _setup(self, orders, orderT, names):
        po_list = [orderT(_) for _ in product(*list(range(o + 1) for o in orders))]

        self.orders = orderT(orders)
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
            v1 = np.multiply.reduce([pow(vv[k], po[k])
                                     for k in self.names])
            v += p*v1

        return v

    def get_array(self, vv):
        v_list = []
        for po in self.po_list:
            v1 = np.multiply.reduce([pow(vv[k], po[k])
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
        orderT = my_namedtuple("order_" + "_".join(names), names)

        self._setup(orders, orderT, names)

    def _get_frozen_p(self, k_survived):
        p = NdPolyNamed([self.orders[_k] for _k in k_survived], k_survived)
        return p


#         T = namedtuple("order_"+"_".join(_), _)

# fix_k = "order"
# fix_value = 80

# _ = tuple(k for k in names if k != fix_k)
# T = namedtuple("order_"+"_".join(_), _)

# po_list = [T(*_) for _ in product([0, 1, 2], [0, 1, 2])]
# poo = dict((k, []) for k in po_list)

# for sol0, po in zip(sol, po_list0):
#     _ = (p for k, p in zip(names, po) if k != fix_k)
#     k = T(*_)
#     #print po, T(*_)
#     oo = getattr(po, fix_k)
#     poo[k].append(sol0 * pow(fix_value, oo))

# sol1 = [np.sum(poo[po]) for po in po_list]


# #def test():
# if 1:
#     p0 = NdPoly([2, 2, 2])
#     params0 = np.arange(1, 28)
#     x0 = p0.multiply([2, 3, 5], params0)
#     p1, params1 = p0.freeze(0, 2, params0)
#     x1 = p1.multiply([3, 5], params1)
#     assert x0 == x1
#     p2, params2 = p0.freeze(1, 3, params0)
#     x2 = p2.multiply([2, 5], params2)
#     assert x0 == x2

# if 1:
#     names = ["i1", "i2", "i3"]
#     p0 = NdPolyNamed([2, 2, 2], names)
#     params0 = np.arange(1, 28)
#     x0 = p0.multiply(dict(zip(names, [2, 3, 5])), params0)
