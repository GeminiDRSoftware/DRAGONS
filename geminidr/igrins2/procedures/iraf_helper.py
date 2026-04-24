import astropy.io.fits as pyfits

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
        c = pyfits.Card(k, v)
        cards.append(c)

    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        #print k, v
        c = pyfits.Card(k, v)
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

    f= pyfits.open("outdata/20140525/SDCK_20140525_0016.spec.fits")
