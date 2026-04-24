# import astropy.io.fits as pyfits
import numpy as np
import pandas as pd

from astropy.table import Table
import astrodata
import igrins_instruments

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


if True:
        from operator import itemgetter
        from scipy.interpolate import interp1d
        from igrinsdr.igrins.primitives_igrins import get_ref_path

        ad = adinputs[0]

        # prepare line fitting. Reference lines are read and we will define
        # _fit function which will fit lines given a list of spectrum. _fit
        # will be applied to spectra of different slit positions.

        wvlsol0 = ad[0].WVLSOL0
        orders, wvlsol = wvlsol0["orders"], wvlsol0["wavelengths"]
        wvlsol_by_order = dict(zip(orders, wvlsol))

        ref_file = get_ref_path(band, "ref_lines_oh") # "ref_lines_oh.fits"
        tbl = Table.read(ref_file.open("rb"), format="fits") # "ref_lines_oh.fits"
        df_ref_data0 = tbl.to_pandas()

        from scipy.interpolate import interp1d
        x = np.arange(2048)
        # for each order, add pixel coordinate from the initial wvlsol
        for order, grouped in df_ref_data0.groupby("order"):
            wvl = wvlsol_by_order.get(order, None)
            if wvl is not None:
                knots = interp1d(wvl, x,
                                 bounds_error=False, assume_sorted=True, fill_value=np.nan)
                df_ref_data0.loc[grouped.index, "pixel"] = knots(grouped["um"])

        # flags groups that any of the line in the group has a pixel value of nan.
        msk = df_ref_data0.groupby("gid")["pixel"].apply(lambda pixels:
                                                        np.all(np.isfinite(pixels)))
        # msk has an index of "gid". We will filter the dataframe using this mask.
        # Note that there can be multiple rows wit same gid, and indexing with mask
        # gives a warning of

        # Boolean Series key will be reindexed to match DataFrame index

        # FIXME check if there is a better way of doing this.
        df_ref_data = df_ref_data0.set_index("gid")[msk].reset_index()

        # The filtered df_ref_data should only have valid pixels.


        def _fit(df_ref_data, spec_by_order):
            # we prepare a dataframe index of (order, gid)
            grouped = df_ref_data.groupby(["order", "gid"])
            df_fit = pd.DataFrame(dict(initial_mean_pixel=grouped["pixel"].mean(),
                                       wavelength=grouped["um"].mean()))

            fitted_line_location = pd.Series(index=df_ref_data.index) # initially set to nan

            # For each group, we fit the sliced data with multiple gaussian.
            for (o, gid), grp in grouped:
                if (s := spec_by_order.get(o, None)) is not None:
                    r = fit_gaussian_group(x, s, grp["pixel"])
                    # add column for the fit parameter
                    df_fit.loc[(o, gid), ["shift", "sigma", "height", "baseline"]] = r[0]
                    # add column for fitted pixel position
                    df_fit.loc[(o, gid), "fitted_pixel"] = df_fit.loc[(o, gid), "initial_mean_pixel"] + r[0][0]
                    fitted_line_location[grp.index] = grp["pixel"] + r[0][0]

            return df_fit, fitted_line_location

        # fit lines in the spectrum of the slit center
        multi_spec = ad[0].SPEC1D_MULTI
        slit_centers = multi_spec["slit_centers"][0].astype("float32")
        i_slit_center = len(slit_centers) // 2

        spec_data = multi_spec["multispec"][:, i_slit_center, :]
        spec_by_order = dict(zip(multi_spec["orders"], spec_data))

        df_fit_list = []
        df_fit, fitted_line_location0 = _fit(df_ref_data, spec_by_order)
        df_fit_list.append((slit_centers[i_slit_center], df_fit))

        # Now we do lower and upper part of the slit
        for i_range in [range(0, i_slit_center)[::-1], # lower part of the slit
                        range(i_slit_center+1, len(slit_centers)) # upper part of the slit
                        ]:
            # for the start of upper and lower parts, the initial location of
            # lines are from the slit center.
            fitted_line_location = fitted_line_location0
            for i in i_range:
                # we update the pixel location from the previous fit
                df_ref_data_updated = df_ref_data.copy(deep=False)
                df_ref_data_updated["pixel"] = fitted_line_location

                spec_data = multi_spec["multispec"][:, i, :]
                spec_by_order = dict(zip(multi_spec["orders"], spec_data))

                df_fit, fitted_line_location = _fit(df_ref_data, spec_by_order)
                df_fit_list.append((slit_centers[i], df_fit))


        df_fit_list.sort(key=itemgetter(0))
        df_fit_oh = pd.concat([df_fit for _, df_fit in df_fit_list],
                              keys=[c for c, _ in df_fit_list],
                              names=["slit_center"],
                              axis=0)

        df_fit_master = pd.concat([df_fit_oh],
                                  keys=["oh"],
                                  names=["kind"],
                                  axis=0)


        tbl = Table.from_pandas(df_fit_master.reset_index())

        tbl.write("fit_master.fits", overwrite=True)
