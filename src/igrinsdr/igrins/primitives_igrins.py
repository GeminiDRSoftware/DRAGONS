#
#                                                                       DRAGONS
#
#                                                         primitives_igrins.py
# ------------------------------------------------------------------------------

import numpy as np

from astropy.io import fits
from astropy.table import Table

import astrodata
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.primitives_gemini import Gemini
from geminidr.core.primitives_nearIR import NearIR
from geminidr.gemini.lookups import DQ_definitions as DQ

from . import parameters_igrins

from .lookups import timestamp_keywords as igrins_stamps

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

from .procedures.procedure_dark import (make_guard_n_bg_subtracted_images,
                                        estimate_amp_wise_noise)

from .procedures.trace_flat import trace_flat_edges, table_to_poly
from .procedures.iter_order import iter_order


@parameter_override
class Igrins(Gemini, NearIR):
    """
    This class inherits from the level above.  Any primitives specific
    to IGRINS can go here.
    """

    tagset = {"GEMINI", "IGRINS"}

    def __init__(self, adinputs, **kwargs):
        super(Igrins, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_igrins)
        # Add IGRINS specific timestamp keywords
        self.timestamp_keys.update(igrins_stamps.timestamp_keys)

    def selectFrame(self, adinputs=None, **params):
        """Filter the adinputs by its FRMTYPE value in the header.
        """
        frmtype = params["frmtype"]
        adoutputs = [ad for ad in adinputs
                     if frmtype in ad.hdr['FRMTYPE']]
        return adoutputs

    def streamPatternCorrected(self, adinputs=None, **params):
        """
        make images with Readout pattern corrected. And add them to streams.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        rpc_mode = params.get("rpc_mode")
        assert rpc_mode == "full"
        # FIXME: only 'full' mode is supported for now, which will create
        # images using the methods of ['guard', 'level2', 'level3']

        dlist = [ad[0].data for ad in adinputs]
        hdu_list = make_guard_n_bg_subtracted_images(dlist,
                                                     rpc_mode=rpc_mode,
                                                     bias_mask=None,
                                                     log=log)
        for (name, dlist) in hdu_list:
            # name: the name of the correction method applied. One of ["GUARD",
            # "LEVEL2", "LEVEL3"]
            # dlist : list of numpy images
            adoutputs = []
            for ad0, d in zip(adinputs, dlist):
                # we create new astrodata object based on the input's header.
                hdu = fits.ImageHDU(data=d, header=ad0[0].hdr,
                                    name='SCI')
                ad = astrodata.create(ad0.phu, [hdu])
                gt.mark_history(ad, primname=self.myself(),
                                keyword="RPC")

                adoutputs.append(ad)

            self.streams[f"RPC_{name}"] = adoutputs

        return adinputs

    def estimateNoise(self, adinputs=None, **params):
        """Estimate the noise characteriscs for images in each streams. The resulting
        table is added to a 'ESTIMATED_NOISE' stream
        """

        # filenames that will be used in the table.
        filenames = [ad.filename for ad in adinputs]

        kdlist = [(k[4:], [ad[0].data for ad in adlist])
                  for k, adlist in self.streams.items()
                  if k.startswith("RPC_")]

        df = estimate_amp_wise_noise(kdlist, filenames=filenames)
        # df : pandas datafrome object.

        # Convert it to astropy.Table and then to an astrodata object.
        tbl = Table.from_pandas(df)
        phu = fits.PrimaryHDU()
        ad = astrodata.create(phu)

        astrodata.add_header_to_table(tbl)
        ad.EST_NOISE = tbl
        # ad.append(tbl, name='EST_NOISE')

        self.streams["ESTIMATED_NOISE"] = [ad]

        return adinputs

    def selectStream(self, adinputs=None, **params):
        stream_name = params["stream_name"]
        return self.streams[f"RPC_{stream_name}"]

    def addNoiseTable(self, adinputs=None, **params):
        """
        The table from the 'EST_NOISE' stream will be appended to the input.
        """
        # adinputs should contain a single ad of stacked dark. We attach table
        # to the stacked dark.

        ad = adinputs[0]

        ad_noise_table = self.streams["ESTIMATED_NOISE"][0]
        del self.streams["ESTIMATED_NOISE"]

        ad.EST_NOISE = ad_noise_table.EST_NOISE
        # ad.append(ad_noise_table.EST_NOISE, name="EST_NOISE")

        return adinputs

    def setSuffix(self, adinputs=None, **params):
        suffix = params["suffix"]

        # Doing this also makes the output files saved.
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                           primname=self.myself(),
                                           keyword="NOISETABLE")

        return adinputs

    def someStuff(self, adinputs=None, **params):
        """
        Write message to screen.  Test primitive.

        Parameters
        ----------
        adinputs
        params

        Returns
        -------

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.status('I see '+ad.filename)

            gt.mark_history(ad, primname=self.myself(), keyword="TEST")
            ad.update_filename(suffix=params['suffix'], strip=True)

        return adinputs


    @staticmethod
    def _has_valid_extensions(ad):
        """ Check that the AD has a valid number of extensions. """

        # this needs to be updated at appropriate.
        return len(ad) in [1]

    def determineSlitEdges(self, adinputs=None, **params):

        ad = adinputs[0]

        # ll = trace_flat_edges(ad[0].data)
        # print(ad.info())
        # tbl = Table(ll)

        # ad.SLITEDGE = tbl

        for ext in ad:
            ll = trace_flat_edges(ext.data)
            tbl = Table(ll)

            ext.SLITEDGE = tbl

        return adinputs

    def maskBeyondSlit(self, adinputs=None, **params):

        ad = adinputs[0]

        for ext in ad:
            tbl = ext.SLITEDGE

            pp = table_to_poly(tbl)

            mask = np.zeros((2048, 2048), dtype=bool)
            for o, sl, m in iter_order(pp):
                mask[sl][m] = True

            ext.mask |= ~mask * DQ.unilluminated

        return adinputs

    def normalizeFlat(self, adinputs=None, **params):

        ad = adinputs[0]

        for ext in ad:
            tbl = ext.SLITEDGE

            pp = table_to_poly(tbl)

            ext.FLAT_ORIGINAL = ext.data.copy()
            d = ext.data
            dq_mask = (ext.mask & DQ.unilluminated).astype(bool)
            d[dq_mask] = np.nan

            # Very primitive normarlization. Should be improved.
            for o, sl, m in iter_order(pp):
                dn = np.ma.array(d[sl], mask=~m).filled(np.nan)
                s = np.nanmedian(dn,
                                 axis=0)
                d[sl][m] = (dn / s)[m]

        return adinputs

    def fixIgrinsHeader(self, adinputs, **params):
        ad = adinputs[0]

        for ad in adinputs:
            for ext in ad:
                for desc in ('saturation_level', 'non_linear_level'):
                    kw = ad._keyword_for(desc)
                    if kw not in ext.hdr:
                        ext.hdr[kw] = (1.e5, "Test")
                        # print ("FIX", kw, ext.hdr.comments[kw])


        return adinputs

