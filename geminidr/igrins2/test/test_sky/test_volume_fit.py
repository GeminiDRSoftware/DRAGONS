# import astropy.io.fits as pyfits
import numpy as np
import pandas as pd

from astropy.table import Table
import astrodata, gemini_instruments

if True:
        linefit = Table.read("fit_master.fits", format="fits")

        colnames = linefit.colnames

        df = linefit.to_pandas().rename(columns=dict(fitted_pixel="pixel"))

        # we now calculate offset of pixels from the lines at the central slit
        slit_centers = sorted(df["slit_center"].unique())
        i_slit_center = len(slit_centers) // 2
        sc_center = slit_centers[i_slit_center]

        # We populate the "pixel0" column with the pixel value of central slit,
        # then subtract pixel0 from pixel. FIXME be a better way of doing this?

        # FIXME make sure gid is unique regardless of kind
        dft = df.set_index(["slit_center", "gid"])
        pixel0 = dft.loc[sc_center, "pixel"]

        # FIXME it coule be better to simply using the numpy operation instead
        # of reindexing.
        dft["pixel0"] = pixel0.reindex(dft.index, level=1)
        # dft["pixel0"] = np.tile(pixel0.values, len(slit_centers)) # FIXME is this safe?
        dft["offset"] = dft["pixel"] - dft["pixel0"]

        # FILTER_POINTS

        # We will drop outliers from both side.
        # FIXME Can we simply do the fitting and drop the outliers?

        # index_names = ["kind", "order", "wavelength"]
        # dfs = df.reset_index().set_index(index_names)[["slit_center", "pixel", "offset"]]

        # ss0 = df.groupby("pixel0")["offset"]
        ss0 = dft.groupby("gid")["offset"]
        ss0_std = ss0.std()
        # ss0_std = ss0.transform(np.std)

        ss = ss0.std()
        drop = 0.1
        vmin = np.percentile(ss, 100*drop)
        vmax = np.percentile(ss, 100*(1 - drop))

        msk = (ss0_std > vmin) & (ss0_std < vmax)

        mskk = msk.reindex(dft.index, level=1)
        dd = dft.loc[mskk].reset_index()

        dd.to_json("points.json")

