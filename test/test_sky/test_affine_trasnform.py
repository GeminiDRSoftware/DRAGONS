# import astropy.io.fits as pyfits
import numpy as np

from astropy.table import Table
import astrodata
import igrins_instruments

from igrinsdr.igrins.primitives_igrins import get_ref_spectra, get_ref_data

# from igrinsdr.igrins.procedures.procedures_register import _get_offset_transform_between_2spec

from igrinsdr.igrins.procedures.identified_lines import IdentifiedLines
from igrinsdr.igrins.procedures.apertures import Apertures
from igrinsdr.igrins.procedures.echellogram import Echellogram
# from igrinsdr.igrins.primitives_igrins import _get_wavelength_solutions

# fn = "N20240429S0204_K.fits"
band = "H"
# band = "K"

fnroot = "N20240429S0204"

# fn = f"{fnroot}_{band}.fits"
# ad = astrodata.open(fn)
# adlist = [ad]

fnout = f"../../test_i2/{fnroot}_{band}_arc.fits"
adout = astrodata.open(fnout)
adinputs = [adout]


# from igrinsdr.igrins.primitives_igrins import IdentifiedLines, getInitialWvlsol

if True:
    from igrinsdr.igrins.procedures.procedures_register import get_offset_transform_between_two_specs

    # snippets from identifyLines

    ad = adinputs[0]
    ext = ad[0]

    band = ext.band() # phu["BAND"]
    orders_ref, s_list_ref = get_ref_spectra(band)

    tgt_spec = ext.SPEC1D

    ref_spec = get_ref_data(band, "ref_spec")

    # ref_spec = ref_data["ref_spec"]

    tr_ref_to_tgt = get_offset_transform_between_two_specs(ref_spec, tgt_spec)

    # l = ref_data["identified_lines_v0"]
    l = get_ref_data(band, "identified_lines_v0")
    identified_lines_ref = IdentifiedLines(l)

    identified_lines_tgt = identified_lines_ref.reidentify_specs(tgt_spec["orders"],
                                                                 tgt_spec["specs"],
                                                                 tr_ref_to_tgt,
                                                                 offset_threshold=3)

    df_ref = identified_lines_ref.get_df()
    df_tgt = identified_lines_tgt.get_df()

    # See if we can do some tests.

    fig, ax = plt.subplots(1, 1, num=1, clear=True)
    msk = (df_tgt["xpos"] > 0) & (df_ref["xpos"] > 0)
    plt.plot(df_ref[msk]["xpos"], df_ref[msk]["order"], "o")
    plt.plot(df_tgt[msk]["xpos"], df_tgt[msk]["order"], "o")




if True:
    from igrinsdr.igrins.primitives_igrins import get_xy_of_ref_n_tgt, get_wvlsol_from_transformed_echellogram
    from igrinsdr.igrins.procedures.fit_affine import fit_affine_clip

    # snippets from getInitialWvlsol

    ad = adinputs[0]
    ext = ad[0]
    tgt_spec = ext.SPEC1D

    df_identified_lines = ad[0].LINEID.to_pandas()

    ap = Apertures(ad[0].SLITEDGE)

    band = ext.band() # phu["BAND"]

    echellogram_data = get_ref_data(band, "echellogram_data")
    echellogram = Echellogram.from_dict(echellogram_data)

    # We may use the xpos and ypos to fit the wavelength solution. However,
    # it is assumed that we do not have many lines to cover whole detector
    # area thus doing that will may give unstable wavelength solution.
    # Therefore, we fit affine transform from reference to the target,
    # transform each orders' x, y position of echellogram (length of 2048;
    # y position is not actually used), and then fit that to derive a new
    # wavelength solution.

    dfout = get_xy_of_ref_n_tgt(df_identified_lines, ap, echellogram)
    xy_list_ref = dfout[["xpos0", "ypos0"]].values # from reference echellogram
    xy_list_tgt = dfout[["xpos", "ypos"]].values # idntified from the target.

    # find the affine transform.
    affine_tr, mm = fit_affine_clip(xy_list_ref, xy_list_tgt)

    affine_tr_matrix = affine_tr.get_matrix()

    # we now transform the echellogram with the affine transform.

    orders = tgt_spec["orders"].data # .data to convert Table column to numpy array
    wvl_sol = get_wvlsol_from_transformed_echellogram(echellogram,
                                                      affine_tr_matrix,
                                                      orders)

    # some test

    fig, ax = plt.subplots(1, 1, num=2, clear=True)

    for z in echellogram.zdata.values():
        ax.plot(z.wvl, "0.8", lw=4)

    for w in wvl_sol:
        ax.plot(w, "r", ls=":")
