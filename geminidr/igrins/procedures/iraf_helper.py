import numpy as np

from astropy.io import fits
import json

from .astropy_poly_helper import deserialize_poly_model


default_header_str = """WCSDIM  =                    3
CTYPE1  = 'MULTISPE'
CTYPE2  = 'MULTISPE'
CTYPE3  = 'LINEAR  '
LTM1_1  =                   1.
LTM2_2  =                   1.
CD1_1   =                   1.
CD2_2   =                   1.
CD3_3   =                   1.
LTM3_3  =                   1.
WAT0_001= 'system=multispec'
WAT1_001= 'wtype=multispec label=Wavelength units=angstroms'
WAT3_001= 'wtype=linear'
BANDID1 = 'spectrum - background median, weights variance, clean no'
""".strip().split("\n")

def get_wat_cards(fit_results_tbl):

    fit_results_map = dict(zip(fit_results_tbl["key"], fit_results_tbl["encoded"]))
    fitted_model_encoded = fit_results_map["fitted_model"]
    orders = json.loads(fit_results_map["orders"])
    modeul_name, class_name, serialized = json.loads(fitted_model_encoded)

    p = deserialize_poly_model(modeul_name, class_name, serialized)

    # save as WAT fits header
    xx = np.arange(0, 2048)
    xx_plus1 = np.arange(1, 2048+1)

    from astropy.modeling import models, fitting

    # We convert 2d chebyshev solution to a seriese of 1d
    # chebyshev.  For now, use naive (and inefficient)
    # approach of refitting the solution with 1d. Should be
    # reimplemented.

    p1d_list = []
    for o in orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o * 1.e4  # um to angstrom

        p_init1d = models.Chebyshev1D(domain=[1, 2048],
                                      degree=p.x_degree)
        fit_p1d = fitting.LinearLSQFitter()
        p1d = fit_p1d(p_init1d, xx_plus1, wvl)
        p1d_list.append(p1d)

    wat_list = get_wat_spec(orders, p1d_list)

    cards = [fits.Card.fromstring(l.strip())
             for l in default_header_str]

    wat = "wtype=multispec " + " ".join(wat_list)
    char_per_line = 68
    num_line, remainder = divmod(len(wat), char_per_line)
    for i in range(num_line):
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:char_per_line*(i+1)]
        c = fits.Card(k, v)
        cards.append(c)

    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        c = fits.Card(k, v)
        cards.append(c)

    return cards


def get_wat_header(wat_table, wavelength_increasing_order=False):

    cards = [fits.Card.fromstring(s) for s in wat_table['cards']]
    header = fits.Header(cards)

    # hdu = obsset.load_resource_sci_hdu_for("wvlsol_fits")
    if wavelength_increasing_order:
        header = invert_order(header)

        def convert_data(d):
            return d[::-1]
    else:

        def convert_data(d):
            return d

    return header, convert_data

def get_wat_spec(orders, wvl_sol):
    """
    WAT header
    wvl_sol should be a list of Chebyshev polynomials.
    """

    # specN = ap beam dtype w1 dw nw z aplow aphigh
    specN_tmpl = "{ap} {beam} {dtype} {w1} {dw} {nw} {z} {aplow} {aphigh}"
    function_i_tmpl = "{wt_i} {w0_i} {ftype_i} {parameters} {coefficients}"
    specN_list = []
    from itertools import count
    for ap_num, o, wsol in zip(count(1), orders, wvl_sol):
        specN = "spec%d" % ap_num

        d = dict(ap=ap_num,
                 beam=o,
                 dtype=2,
                 w1=wsol(0),
                 dw=(wsol(2047)-wsol(0))/2048.,
                 nw=2048,
                 z=0.,
                 aplow=0.,
                 aphigh=1.
                 )
        specN_str = specN_tmpl.format(**d)

        param_d = dict(c_order=wsol.degree+1, # degree in iraf def starts from 1
                       pmin=wsol.domain[0],
                       pmax=wsol.domain[1])
        d = dict(wt_i=1.,
                 w0_i=0.,
                 ftype_i=1, # chebyshev(1), legendre(2), etc.
                 parameters="{c_order} {pmin} {pmax}".format(**param_d),
                 coefficients=" ".join(map(str, wsol.parameters)))

        function_i = function_i_tmpl.format(**d)

        s = '%s = "%s %s"' % (specN, specN_str, function_i)

        specN_list.append(s)

    return specN_list


def get_wat2_spec_cards(wat_list):

    wat = "wtype=multispec " + " ".join(wat_list)
    char_per_line = 68
    num_line, remainder = divmod(len(wat), char_per_line)
    cards = []
    for i in range(num_line):
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:char_per_line*(i+1)]
        #print k, v
        c = fits.Card(k, v)
        cards.append(c)

    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        #print k, v
        c = fits.Card(k, v)
        cards.append(c)

    return cards


def invert_order(header):
    """
    """

    import re
    p = re.compile(r"\s*spec\d+\s*")

    new_cards = []

    wat_s_list = []
    for c in header.cards:
        if c.keyword.startswith("WAT2"):
            wat_s_list.append((c.keyword, c.value))
        else:
            new_cards.append(c)

    wat_s_list.sort()

    def pad68(s):
        return s+" "*max(68 - len(s), 0)

    wat_str = "".join(pad68(v) for k, v in wat_s_list)
    wat_str = wat_str.replace("wtype=multispec","").strip()

    wat_spec_list = p.split(wat_str)

    wat_list_r = ["spec%d %s" % (i+1, s) for i, s in enumerate(wat_spec_list[::-1]) if s.strip()]


    cards = get_wat2_spec_cards(wat_list_r)

    new_cards.extend(cards)

    return type(header)(new_cards)



if __name__ == "__main__":

    import numpy as np
    xxx = np.linspace(1, 2048, 100)
    yyy = xxx**4

    from astropy.modeling import models, fitting
    p_init = models.Chebyshev1D(domain=[xxx[0], xxx[-1]],
                                            degree=4)
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, xxx, yyy)

    wat_list = get_wat_spec([111], [p])

    f= fits.open("outdata/20140525/SDCK_20140525_0016.spec.fits")
