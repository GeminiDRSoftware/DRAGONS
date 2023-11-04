#
#                                                                  gemini_python
#
#                                                            primitives_ghost.py
# ------------------------------------------------------------------------------
from geminidr.gemini.primitives_gemini import Gemini
from geminidr.core.primitives_ccd import CCD
from .primitives_calibdb_ghost import CalibDBGHOST

from . import parameters_ghost

from .lookups import keyword_comments, timestamp_keywords as ghost_stamps

from recipe_system.utils.decorators import parameter_override

import re
import astrodata
from gempy.gemini import gemini_tools as gt
import numpy as np
# ------------------------------------------------------------------------------
_HDR_SIZE_REGEX = re.compile(r'^\[(?P<x1>[0-9]*)\:'
                             r'(?P<x2>[0-9]*),'
                             r'(?P<y1>[0-9]*)\:'
                             r'(?P<y2>[0-9]*)\]$')


def filename_updater(ad, **kwargs):
    origname = ad.filename
    ad.update_filename(**kwargs)
    rv = ad.filename
    ad.filename = origname
    return rv


@parameter_override
class GHOST(Gemini, CCD, CalibDBGHOST):
    """
    Top-level primitives for handling GHOST data

    The primitives in this class are applicable to all flavours of GHOST data.
    All other GHOST primitive classes inherit from this class.
    """
    tagset = set()  # Cannot be assigned as a class

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.ghost.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_ghost)
        # Add GHOST-specific timestamp keywords
        self.timestamp_keys.update(ghost_stamps.timestamp_keys)
        self.keyword_comments.update(keyword_comments.keyword_comments)

    @staticmethod
    def _has_valid_extensions(ad):
        return len(ad) > 0

    def old_rebin_ghost_ad(self, ad, xb, yb):
        """
        Internal helper function to re-bin GHOST data.

        .. note::
            This function is *not* a primitive. It is designed to be called
            internally by public primitives.

        This function should be used for all re-binning procedures on
        AstroData objects that will be saved during reduction. It is designed
        to handle the correct adjustment of the relevant header keywords.

        Binning is done by simply adding together the data pixels in the
        input AstroData object.

        If the input ad contains a variance plane, the re-binned variance
        plane is computed by summing over the binned variance pixels in
        quadrature. If a mask plane is present, the re-binned mask plane is
        computed by sequentially bitwise_or combining the input mask pixels.

        .. note::
            This function has been included within the GHOST primitive class
            mostly so logging is consistent. Otherwise, it could be defined as
            a @staticmethod (or just exist outside the class completely).

        Parameters
        ----------
        ad : :class:`astrodata.AstroData`
            AstroData object to be re-binned. Each extension of the object
            will be rebinned separately. A :any:`ValueError` will be thrown
            if the object's extensions are found to have different binning
            modes to one another.
        xb : :obj:`int`
            x-binning
        yb : :obj:`int`
            y-binning

        Returns
        -------
        ad : :any:`astrodata.AstroData` object
            Input AstroData object, re-binned to the requested format
        """
        log = self.log

        # Input checking
        if not isinstance(ad, astrodata.AstroData):
            raise ValueError('ad is not a valid AstroData instance')

        xbin, ybin = int(xb), int(yb)
        xrebin, yrebin = xbin // ad.detector_x_bin(), ybin // ad.detector_y_bin()
        if xrebin < 1 or yrebin < 1:
            raise ValueError('Attempt to rebin to less coarse binning')
        #for ext in ad:
        #    if ext.hdr.get('CCDSUM') != '1 1':
        #        raise ValueError(
        #            'Cannot re-bin data that has already been binned')

        # Re-binning
        log.stdinfo('Re-binning %s' % ad.filename)
        for ext in ad:
            orig_shape = ext.data.shape
            # Do the re-binning
            # Re-bin data
            binned_array = ext.data.reshape(
                int(ext.data.shape[0] / yrebin), yrebin,
                int(ext.data.shape[1] / xrebin), xrebin
            ).sum(axis=1).sum(axis=2)

            reset_kwargs = {'check': True, }

            # Re-bin variance
            # These are summed in quadrature
            if ext.variance is not None:
                #binned_var = ext.variance ** 2
                binned_var = ext.variance.reshape(
                    int(ext.variance.shape[0] / yrebin), yrebin,
                    int(ext.variance.shape[1] / xrebin), xrebin
                ).sum(axis=1).sum(axis=2)
                #binned_var = np.sqrt(binned_var)
                reset_kwargs['variance'] = binned_var

            # Re-bin mask
            # This can't be done in an easy one-liner - numpy bitwise_and
            # is designed to combine two distinct arrays, not combine a
            # unitary array a la, e.g. sum
            if ext.mask is not None:
                reshaped_mask = ext.mask.reshape(
                    int(ext.mask.shape[0] / yrebin), yrebin,
                    int(ext.mask.shape[1] / xrebin), xrebin
                )
                binned_mask_r = reshaped_mask[:, 0, :, :]
                for i in range(1, yrebin):
                    binned_mask_r = np.bitwise_or(binned_mask_r,
                                                  reshaped_mask[:, i, :, :])
                binned_mask = binned_mask_r[:, :, 0]
                for j in range(1, xrebin):
                    binned_mask = np.bitwise_or(binned_mask,
                                                binned_mask_r[:, :, j])
                reset_kwargs['mask'] = binned_mask

            # Alter the data values (do this all together in case one of the
            # calculations above bombs
            # ext.data = binned_array
            # ext.variance = binned_var
            # ext.mask = binned_mask
            ext.reset(binned_array, **reset_kwargs)

            # Rebin other extensions (e.g., PIXELMODEL)
            for other_name, other_data in ext.nddata.meta['other'].items():
                try:
                    rebin_this = other_data.shape == orig_shape
                except AttributeError:
                    continue
                if rebin_this:
                    rebinned_other_data = other_data.reshape(
                        orig_shape[0] // yrebin, yrebin,
                        orig_shape[1] // xrebin, xrebin
                    ).sum(axis=1).sum(axis=2)
                    setattr(ext, other_name, rebinned_other_data)

            # Update header values
            ext.hdr.set('CCDSUM',
                        value=f'{xbin} {ybin}',
                        comment=f'Re-binned to {xbin} {ybin}')

            old_datasec = ext.hdr.get('DATASEC')
            if old_datasec:
                datasec_values = _HDR_SIZE_REGEX.match(old_datasec)
                ext.hdr.set('DATASEC',
                            value='[%d:%d,%d:%d]' %
                                  (max(int(datasec_values.group('x1')) / xrebin,
                                       1),
                                   max(int(datasec_values.group('x2')) / xrebin,
                                       1),
                                   max(int(datasec_values.group('y1')) / yrebin,
                                       1),
                                   max(int(datasec_values.group('y2')) / yrebin,
                                       1),
                                   ),
                            comment=f'Re-binned to {xbin}x{ybin}',
                            )
            old_trimsec = ext.hdr.get('TRIMSEC')
            if old_trimsec:
                trimsec_values = _HDR_SIZE_REGEX.match(old_trimsec)
                ext.hdr.set('TRIMSEC',
                            value='[%d:%d,%d:%d]' %
                                  (max(int(trimsec_values.group('x1')) / xrebin,
                                       1),
                                   max(int(trimsec_values.group('x2')) / xrebin,
                                       1),
                                   max(int(trimsec_values.group('y1')) / yrebin,
                                       1),
                                   max(int(trimsec_values.group('y2')) / yrebin,
                                       1),
                                   ),
                            comment=f'Re-binned to {xbin}x{ybin}',
                            )

            old_ampsize = ext.hdr.get('AMPSIZE')
            if old_ampsize:
                ampsize_values = _HDR_SIZE_REGEX.match(old_ampsize)
                ext.hdr.set('AMPSIZE',
                            value='[%d:%d,%d:%d]' %
                                  (max(int(ampsize_values.group('x1')) / xrebin,
                                       1),
                                   max(int(ampsize_values.group('x2')) / xrebin,
                                       1),
                                   max(int(ampsize_values.group('y1')) / yrebin,
                                       1),
                                   max(int(ampsize_values.group('y2')) / yrebin,
                                       1),
                                   ),
                            comment=f'Re-binned to {xbin}x{ybin}',
                            )

        log.stdinfo('Re-binning complete')

        return ad
