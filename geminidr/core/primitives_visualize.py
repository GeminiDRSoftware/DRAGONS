#
#                                                                  gemini_python
#
#                                                        primitives_visualize.py
# ------------------------------------------------------------------------------
import json
import numpy as np
import time
import inspect
import urllib.request

from copy import deepcopy
from importlib import import_module
from contextlib import suppress

from gempy.utils import logutils
from gempy.gemini import gemini_tools as gt
from gempy import numdisplay as nd
from gempy.library import transform
from gempy.display.numdisplay_tools import make_overlay_mask

from astropy.modeling import models
from gwcs.coordinate_frames import Frame2D
from gwcs.wcs import WCS as gWCS

from geminidr.gemini.lookups import DQ_definitions as DQ
from gemini_instruments.gmos.pixel_functions import get_bias_level

from geminidr import PrimitivesBASE
from . import parameters_visualize

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Visualize(PrimitivesBASE):
    """
    This is the class containing the visualization primitives.
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_visualize)

    def display(self, adinputs=None, **params):
        """
        Displays an image on the ds9 display, using multiple frames if
        there are multiple extensions. Saturated pixels can be displayed
        in red, and overlays can also be shown.

        Parameters
        ----------
        extname: str
            'SCI', 'VAR', or 'DQ': plane to display
        frame: int
            starting frame for display
        ignore: bool
            setting to True turns off the display
        remove_bias: bool
            attempt to subtract bias before displaying?
        threshold: str='auto'/float
            level above which to flag pixels as saturated
        tile: bool
            attempt to tile arrays before displaying?
        zscale: bool
            use zscale algorithm?
        overlay: list
            list of overlays for the display
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # No-op if ignore=True
        if params["ignore"]:
            log.warning("display turned off per user request")
            return

        threshold = params['threshold']
        remove_bias = params.get('remove_bias', False)
        extname = params['extname']
        tile = params['tile']
        zscale = params['zscale']
        overlays = params['debug_overlay']
        frame = params['frame'] if params['frame'] else 1
        overlay_index = 0
        lnd = _localNumDisplay()

        if isinstance(overlays, str):
            try:
                overlays = _read_overlays_from_file(overlays)
            except OSError:
                log.warning(f"Cannot open overlays file {overlays}")
                overlays = None

        for ad in adinputs:
            # Allows elegant break from nested loops
            if frame > 16:
                log.warning("Too many images; only the first 16 are displayed")
                break

            copied = False
            # Threshold and bias make sense only for SCI extension
            if extname != 'SCI':
                threshold = None
                remove_bias = False
            elif threshold == 'None' or threshold == 'none':
                #cannot use .lower above since threshold can be a float
                threshold = None
            elif threshold == 'auto':
                mosaicked = ((ad.phu.get(self.timestamp_keys["mosaicDetectors"])
                              is not None) or
                             (ad.phu.get(self.timestamp_keys["tileArrays"])
                              is not None))
                has_dq = all([ext.mask is not None for ext in ad])
                if not has_dq:
                    if mosaicked:
                        log.warning("Cannot add DQ to mosaicked data; no "
                                    "threshold mask will be applied to "
                                    "{}".format(ad.filename))
                        threshold = None
                    else:
                        # addDQ operates in place so deepcopy to preserve input
                        ad = self.addDQ([deepcopy(ad)])[0]
                        copied = True

            if remove_bias:
                if (ad.phu.get('BIASIM') or ad.phu.get('DARKIM') or
                    ad.phu.get(self.timestamp_keys["subtractOverscan"])):
                    log.fullinfo("Bias level has already been removed from "
                                 "data; no approximate correction will be "
                                 "performed")
                else:
                    try:
                        bias_level = get_bias_level(ad, estimate=False)
                    except NotImplementedError:
                        # For non-GMOS instruments
                        bias_level = None

                    if bias_level is not None:
                        if not copied:
                            ad = deepcopy(ad)  # Leave original untouched!
                            copied = True
                        log.stdinfo("Subtracting approximate bias level from "
                                    "{} for display".format(ad.filename))
                        log.fullinfo("Bias levels used: {}".format(str(bias_level)))
                        for ext, bias in zip(ad, bias_level):
                            ext.subtract(np.float32(bias) if bias is not None
                                         else 0)
                    else:
                        log.warning("Bias level not found for {}; approximate "
                                    "bias will not be removed".format(ad.filename))

            # Check whether data needs to be tiled before displaying
            # Otherwise, flatten all desired extensions into a single list
            num_ext = len(ad)
            if tile and num_ext > 1:
                log.fullinfo("Tiling extensions together before displaying")
                # post-transform metadata is arranged in order of blocks, not
                # slices, so we need to ensure the correct offsets are applied
                # to each slice
                array_info = gt.array_information(ad)
                if copied:
                    ad = self.tileArrays([ad], tile_all=True)[0]
                else:
                    ad = self.tileArrays([deepcopy(ad)], tile_all=True)[0]
                # Logic here in case num_ext overlays sent to be applied to all ADs
                if overlays and len(overlays) + overlay_index >= num_ext:
                    new_overlay = []
                    trans_data = ad.nddata[0].meta.pop("transform")
                    for ext_indices, corner, block in zip(array_info.extensions,
                                                          trans_data["corners"],
                                                          trans_data["block_corners"]):
                        xshift = int(round(corner[1][0]))
                        yshift = int(round(corner[0][0]))
                        for ext_index, b in zip(ext_indices, block):
                            dx, dy = xshift + b[1], yshift + b[0]
                            i = overlay_index + ext_index
                            if overlays[i]:
                                new_overlay.extend([(x+dx, y+dy, r) for x, y, r in overlays[i]])
                    overlays = (overlays[:overlay_index] + (new_overlay,) +
                                overlays[overlay_index+num_ext:])

            # Each extension is an individual display item (if the data have been
            # tiled, then there'll only be one extension per AD, of course)
            for ext in ad:
                if frame > 16:
                    break

                # Squeeze the data to remove any empty dimensions (eg, raw F2 data)
                ext.operate(np.squeeze)

                # Get the data we're going to display. TODO Replace extname with attr?
                try:
                    data = getattr(ext, {'SCI':'data', 'DQ':'mask',
                                         'VAR':'variance'}[extname])
                except KeyError:
                    data = getattr(ext, extname, None)
                dqdata = ext.mask
                if data is None:
                    log.warning("No data to display in {}[{}]".format(ext.filename,
                                                                      extname))
                    continue

                # One-dimensional data (ie, extracted spectra)
                if len(data.shape) == 1:
                    continue

                # Make threshold mask if desired
                masks = []
                mask_colors = []
                if threshold is not None:
                    if threshold != 'auto':
                        satmask = data > threshold
                    else:
                        if dqdata is None:
                            log.warning("No DQ plane found; cannot make "
                                        "threshold mask")
                            satmask = None
                        else:
                            satmask = (dqdata & (DQ.non_linear | DQ.saturated)) > 0
                    if satmask is not None:
                        masks.append(satmask)
                        mask_colors.append(204)

                if overlays:
                    # Could be single overlay, or list. Replicate behaviour of
                    # gt.make_lists (which we can't use because we haven't
                    # made a complete list of displayed extensions at the start
                    # in order to avoid memory bloat)
                    try:
                        overlay = overlays[overlay_index]
                    except TypeError:
                        overlay = overlays
                    except IndexError:
                        if len(overlays) == 1:
                            overlay = overlays[0]
                    try:
                        masks.append(make_overlay_mask(overlay, ext.shape))
                    except Exception:
                        pass
                    else:
                        mask_colors.append(206)
                    overlay_index += 1

                # Define the display name
                if tile and extname == 'SCI':
                    name = ext.filename
                elif tile:
                    name = f'{ext.filename}({extname})'
                else:
                    name = f'{ext.filename}({extname}, extension {ext.id})'

                try:
                    lnd.display(data, name=name, frame=frame, zscale=zscale,
                                bpm=None if extname == 'DQ' else dqdata,
                                quiet=True, masks=masks, mask_colors=mask_colors)
                except OSError:
                    log.warning("ds9 not found; cannot display input")

                frame += 1

                # Print from statistics for flats
                if extname=='SCI' and {'GMOS', 'IMAGE', 'FLAT'}.issubset(ext.tags):
                    good_data = data[dqdata==0] if dqdata is not None else data
                    mean = np.mean(good_data)
                    median = np.median(good_data)
                    log.stdinfo("Twilight flat counts for {}:".format(ext.filename))
                    log.stdinfo("    Mean value:   {:.0f}".format(mean))
                    log.stdinfo("    Median value: {:.0f}".format(median))

        return adinputs

    def inspect(self, adinputs=None, **params):
        """
        Loop through the data, with a pause between the display.

        Parameters
        ----------
        pause: int
            Pause in seconds to add between the display.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        pause = params['pause']

        display_params = self._inherit_params(params, 'display')
        for ad in adinputs:
            self.display([ad], **display_params)
            time.sleep(pause)

        return adinputs

    def mosaicDetectors(self, adinputs=None, **params):
        """
        This primitive does a full mosaic of all the arrays in an AD object.
        An appropriate geometry_conf.py module containing geometric information
        is required.

        The read noise keyword of the output extensions are set to the mean
        of the read noise values returned by the input extensions being tiled.
        The gain keyword of the output is set similarly, with a warning logged
        if this is the case.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files.
        sci_only: bool
            mosaic only SCI image data. Default is False
        order: int (1-5)
            order of spline interpolation
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params['suffix']
        order = params['order']
        attributes = ['data'] if params['sci_only'] else None
        geotable = import_module('.geometry_conf', self.inst_lookups)

        adoutputs = []
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by mosaicDetectors".
                            format(ad.filename))
                adoutputs.append(ad)
                continue

            if len(ad) == 1:
                log.warning("{} has only one extension, so there's nothing "
                            "to mosaic".format(ad.filename))
                adoutputs.append(ad)
                continue

            if not all(np.issubdtype(ext.data.dtype, np.floating) for ext in ad):
                log.warning("Cannot mosaic {} with non-floating point data. "
                            "Use tileArrays instead".format(ad.filename))
                adoutputs.append(ad)
                continue

            transform.add_mosaic_wcs(ad, geotable)

            # If there's an overscan section in the data, this will crash, but
            # we can catch that, trim, and try again. Don't catch anything else
            try:
                ad_out = transform.resample_from_wcs(ad, "mosaic", attributes=attributes,
                                                     order=order, process_objcat=False)
            except ValueError as e:
                if 'data sections' in repr(e):
                    ad = gt.trim_to_data_section(ad, self.keyword_comments)
                    ad_out = transform.resample_from_wcs(ad, "mosaic", attributes=attributes,
                                                         order=order, process_objcat=False)
                else:
                    raise e

            # Update read noise: we assume that all the regions represented
            # by a value have the same number of pixels, so the mean is OK
            with suppress(TypeError):  # some NoneTypes in read_noise
                ad_out[0].hdr[ad._keyword_for('read_noise')] = np.mean(ad.read_noise())
            propagate_gain(ad_out[0], ad.gain())

            ad_out.orig_filename = ad.orig_filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def tileArrays(self, adinputs=None, **params):
        """
        This primitive combines extensions by tiling (no interpolation).
        The array_section() and detector_section() descriptors are used
        to derive the geometry of the tiling, so outside help (from the
        instrument's geometry_conf module) is only required if there are
        multiple arrays being tiled together, as the gaps need to be
        specified.

        If the input AstroData objects still have non-data regions, these
        will not be trimmed. However, the WCS of the final image will
        only be correct for some of the image since extra space has been
        introduced into the image.

        The read noise keyword of the output extensions are set to the mean
        of the read noise values returned by the input extensions being tiled.
        The gain keyword of the output is set similarly, with a warning logged
        if this is the case.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        tile_all: bool
            tile to a single extension, rather than one per array?
            (array=physical detector)
        sci_only: bool
            tile only the data plane?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params['suffix']
        tile_all = params['tile_all']
        attributes = ['data'] if params["sci_only"] else None

        adoutputs = []
        for ad in adinputs:
            if len(ad) == 1:
                log.warning("{} has only one extension, so there's nothing "
                            "to tile".format(ad.filename))
                adoutputs.append(ad)
                continue

            array_info = gt.array_information(ad)
            detshape = array_info.detector_shape
            if not tile_all and set(array_info.array_shapes) == {(1, 1)}:
                log.warning(f"{ad.filename} has nothing to tile, as tile_all=False"
                            " but each array has only one amplifier.")
                adoutputs.append(ad)
                continue

            if tile_all and detshape != (1, 1):  # We need gaps!
                geotable = import_module('.geometry_conf', self.inst_lookups)
                chip_gaps = geotable.tile_gaps[ad.detector_name()]
                try:
                    xgap, ygap = chip_gaps
                except TypeError:  # single number, applies to both
                    xgap = ygap = chip_gaps

            # We need to update data_section so resample_from_wcs doesn't
            # complain. We will also update read_noise to the mean value
            # from the descriptor returns for the extensions in each tile
            kw = ad._keyword_for('data_section')
            kw_readnoise = ad._keyword_for('read_noise')
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()

            # Work out additional shifts required to cope with posisble overscan
            # regions, including those in already-tiled CCDs
            if tile_all:
                yorigins, xorigins = np.rollaxis(np.array(array_info.origins), 1).reshape((2,) + array_info.detector_shape)
                xorigins //= xbin
                yorigins //= ybin
            else:
                yorigins, xorigins = np.zeros((2,) + array_info.detector_shape)
            it_ccd = np.nditer(xorigins, flags=['multi_index'])
            i = 0
            gain_list = []
            read_noise_list = []
            while not it_ccd.finished:
                ccdy, ccdx = it_ccd.multi_index
                shp = array_info.array_shapes[i]
                exts = array_info.extensions[i]
                xshifts = np.zeros(shp, dtype=np.int32)
                yshifts = np.zeros(shp, dtype=np.int32)
                it = np.nditer(np.array(exts).reshape(shp), flags=['multi_index'])
                while not it.finished:
                    iy, ix = it.multi_index
                    ext = ad[int(it[0])]
                    datsec = ext.data_section()
                    if datsec.x1 > 0:
                        xshifts[iy, ix:] += datsec.x1
                    if datsec.x2 < ext.shape[1]:
                        xshifts[iy, ix+1:] += ext.shape[1] - datsec.x2
                    if datsec.y1 > 0:
                        yshifts[iy:, ix] += datsec.y1
                    if datsec.y2 < ext.shape[0]:
                        xshifts[iy+1:, ix] += ext.shape[0] - datsec.y2

                    arrsec = ext.array_section()
                    ext_shift = (models.Shift((arrsec.x1 // xbin - datsec.x1)) &
                                 models.Shift((arrsec.y1 // ybin - datsec.y1)))

                    # We need to have a "tile" Frame to resample to.
                    # We also need to perform the inverse, after the "tile"
                    # frame, of any change we make beforehand.
                    if ext.wcs is None:
                        ext.wcs = gWCS([(Frame2D(name="pixels"), ext_shift),
                                        (Frame2D(name="tile"), None)])
                    elif 'tile' not in ext.wcs.available_frames:
                        #ext.wcs.insert_frame(ext.wcs.input_frame, ext_shift,
                        #                     Frame2D(name="tile"))
                        ext.wcs = gWCS([(ext.wcs.input_frame, ext_shift),
                                        (Frame2D(name="tile"), ext.wcs.pipeline[0].transform)] +
                                       ext.wcs.pipeline[1:])
                        ext.wcs.insert_transform('tile', ext_shift.inverse, after=True)

                    dx, dy = xshifts[iy, ix], yshifts[iy, ix]
                    if tile_all:
                        dx += xorigins[ccdy, ccdx]
                        dy += yorigins[ccdy, ccdx]
                    if dx or dy:  # Don't bother if they're both zero
                        shift_model = models.Shift(dx) & models.Shift(dy)
                        ext.wcs.insert_transform('tile', shift_model, after=False)
                        if ext.wcs.output_frame.name != 'tile':
                            ext.wcs.insert_transform('tile', shift_model.inverse, after=True)

                    # Reset data_section since we're not trimming overscans
                    ext.hdr[kw] = '[1:{},1:{}]'.format(*reversed(ext.shape))
                    read_noise_list.append(ext.read_noise())
                    gain_list.append(ext.gain())
                    it.iternext()

                if tile_all:
                    # We need to shift other arrays if this one is larger than
                    # its expected size due to overscan regions. We've kept
                    # track of shifts we've introduced, but it might also be
                    # the case that we've been sent a previous tile_all=False output
                    if ccdx < detshape[1] - 1:
                        max_xshift = max(xshifts.max(), ext.shape[1] -
                                         (xorigins[ccdy, ccdx+1] - xorigins[ccdy, ccdx]))
                        xorigins[ccdy, ccdx+1:] += max_xshift + xgap // xbin
                    if ccdy < detshape[0] - 1:
                        max_yshift = max(yshifts.max(), ext.shape[0] -
                                         (yorigins[ccdy+1, ccdx] - yorigins[ccdy, ccdx]))
                        yorigins[ccdy+1:, ccdx] += max_yshift + ygap // ybin
                else:
                    if i == 0:
                        ad_out = transform.resample_from_wcs(
                            ad[exts], "tile", attributes=attributes, process_objcat=True)
                    else:
                        ad_out.append(transform.resample_from_wcs(
                            ad[exts], "tile",attributes=attributes, process_objcat=True)[0])
                    with suppress(TypeError):
                        ad_out[-1].hdr[kw_readnoise] = np.mean(read_noise_list)
                    propagate_gain(ad_out[-1], gain_list, report_ext=True)
                    gain_list = []
                    read_noise_list = []

                i += 1
                it_ccd.iternext()

            if tile_all:
                ad_out = transform.resample_from_wcs(ad, "tile", attributes=attributes,
                                                     process_objcat=True)
                with suppress(TypeError):
                    ad_out[0].hdr[kw_readnoise] = np.mean(read_noise_list)
                propagate_gain(ad_out[0], gain_list)

            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.orig_filename = ad.filename
            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)
        return adoutputs

    def plotSpectraForQA(self, adinputs=None, **params):
        """
        Converts AstroData containing extracted spectra into a JSON object. Then,
        push it to the Automated Dataflow Coordination Center (ADCC) Server
        (see notes below) using a POST request.

        This will allow the spectra to be visualized using the QAP SpecViewer
        web browser client.

        Notes
        -----
        This primitive only works if the (ADCC) Server is running locally.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Input data containing extracted spectra.
        url : str
            URL address to the ADCC server.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Data used for plotting.
        """
        url = params["url"]

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        log.stdinfo('Number of input file(s): {}'.format(len(adinputs)))

        spec_packs = []

        for ad in adinputs:

            log.stdinfo('Reading {} aperture(s) from file: {}'.format(
                len(ad), ad.filename))

            timestamp = time.time()

            if 'NCOMBINE' in ad.phu:
                is_stack = ad.phu['NCOMBINE'] > 1
                stack_size = ad.phu['NCOMBINE']
            else:
                is_stack = False
                stack_size = 1

            group_id = ad.group_id().split('_[')[0]
            group_id += ad.group_id().split(']')[1]

            offset = (ad.telescope_y_offset() if ad.dispersion_axis()[0] == 1
                      else ad.telescope_x_offset())
            offset /= ad.pixel_scale()

            spec_pack = {
                "apertures": [],
                "data_label": ad.data_label(),
                "filename": ad.filename,
                "group_id": group_id,
                "is_stack": is_stack,
                "stack_size": stack_size,
                "metadata": [],
                "msgtype": "specjson",
                "offset": offset,
                "pixel_scale": ad.pixel_scale(),
                "program_id": ad.program_id(),
                "timestamp": timestamp,
            }

            for i, ext in enumerate(ad):
                data = ext.data
                stddev = np.sqrt(ext.variance)

                if data.ndim > 1:
                    raise TypeError(
                        "Expected 1D data. Found {:d}D data: {:s}".format(
                            data.ndim, ad.filename))

                wavelength = ext.wcs(np.arange(data.size)).astype(np.float32)
                w_dispersion = np.abs(wavelength[-1] - wavelength[0]) / (data.size - 1)
                w_units = str(ext.wcs.output_frame.unit[0])

                # Create mask for bad data
                mask = np.ma.masked_array(
                    np.zeros_like(wavelength),
                    mask=np.logical_or(
                        ext.mask > 0, np.ma.masked_invalid(data).mask))

                # Retrieve unmasked clump slices
                _slices = [
                    [int(s.start), int(s.stop)]
                    for s in np.ma.clump_unmasked(mask)]

                # Round and convert data/stddev to int to minimize data transfer load
                wavelength = np.round(wavelength, decimals=3)

                _intensity = [
                    [float(w), float(d)]
                    for w, d in zip(wavelength, data)]

                _stddev = [
                    [float(w), float(s)]
                    for w, s in zip(wavelength, stddev)]

                _units = ext.hdr["BUNIT"]

                center = np.round(ext.hdr["XTRACTED"])
                lower = np.round(ext.hdr["XTRACTLO"])
                upper = np.round(ext.hdr["XTRACTHI"])

                aperture = {
                    "center": center,
                    "lower": lower,
                    "upper": upper,
                    "dispersion": w_dispersion,
                    "wavelength_units": w_units,
                    "id": np.round(center + offset),
                    "intensity": _intensity,
                    "intensity_units": _units,
                    "slices": _slices,
                    "stddev": _stddev,
                }

                spec_pack["apertures"].append(aperture)

                log.stdinfo(' Aperture center: {}, Lower: {}, Upper: {}'.format(
                    center, lower, upper))

            spec_packs.append(spec_pack)

            spec_packs_json = json.dumps(spec_packs)

            with open("spec_data.json", 'w') as json_buffer:
                json.dump(spec_packs, json_buffer)

            # Convert string to bytes
            spec_packs_json = spec_packs_json.encode("utf-8")

            try:
                log.stdinfo('Sending data to QA SpecViewer')
                post_request = urllib.request.Request(url)
                postr = urllib.request.urlopen(post_request, spec_packs_json)
                postr.read()
                postr.close()
                log.stdinfo('Success.')

            except urllib.error.URLError:
                log.warning('Failed to connect to ADCC Server.\n'
                            'Make sure it is up and running.')

        return adinputs


def _read_overlays_from_file(filename):
    f = open(filename)
    overlays = []
    this_overlay = []
    for line in f.readlines():
        items = line.strip().split()
        if items:
            try:
                coords = [float(item_) for item_ in items]
            except TypeError:
                pass
            else:
                this_overlay.append(coords if len(coords) == 3 else
                                    [*coords, 0])
        else:
            overlays.append(this_overlay)
            this_overlay = []
    return overlays + this_overlay

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################
class _localNumDisplay(nd.NumDisplay):
    """
    This class overrides the default numdisplay.display function in
    order to implement super-fast overlays.  If this feature can be
    incorporated into numdisplay, this local version should go away.

    mask, if specified, should be a tuple of numpy arrays: the y- and
    x-coordinates of points to be masked.  For example,
    mask = np.where(data>threshold)
    TODO: Can it be an array of booleans, the same size as the data?
    """
    def display(self, pix, name=None, bufname=None, z1=None, z2=None,
                transform=None, bpm=None, zscale=False, contrast=0.25,
                scale=None, masks=None, mask_colors=None,
                offset=None, frame=None, quiet=False):

        """ Displays byte-scaled (UInt8) n to XIMTOOL device.
            This method uses the IIS protocol for displaying the data
            to the image display device, which requires the data to be
            byte-scaled.
            If input is not byte-scaled, it will perform scaling using
            set values/defaults.
        """
        log = logutils.get_logger(__name__)

        #Ensure that the input array 'pix' is a numpy array
        pix = np.array(pix)
        self.z1 = z1
        self.z2 = z2

        # If any of the display parameters are specified here, apply them
        # if z1 or z2 or transform or scale or offset or frame:
        # If zscale=True (like IRAF's display) selected, calculate z1 and z2 from
        # the data, and clear any transform specified in the call
        # Offset and scale are applied to the data and z1,z2,
        # so they have no effect on the display
        if zscale:
            if transform != None:
                if not quiet:
                    log.fullinfo("transform disallowed when zscale=True")
                transform = None
            if bpm is None:
                z1, z2 = nd.zscale.zscale(pix, contrast=contrast)
            else:
                goodpix = pix[bpm==0]
                # Ignore the mask unless a decent number of pixels are "good"
                if len(goodpix) >= 0.01 * np.multiply(*pix.shape):
                    sq_side = int(np.sqrt(len(goodpix)))
                    goodpix = goodpix[:sq_side**2].reshape(sq_side, sq_side)
                    z1, z2 = nd.zscale.zscale(goodpix, contrast=contrast)
                else:
                    z1, z2 = nd.zscale.zscale(pix, contrast=contrast)

        self.set(frame=frame, z1=z1, z2=z2,
                 transform=transform, scale=scale, offset=offset)

        # Initialize the display device
        if not self.view._display or self.view.checkDisplay() is False:
            self.open()
        _d = self.view._display
        self.handle = _d.getHandle()

        # If no user specified values are provided, interrogate the array itself
        # for the full range of pixel values
        if self.z1 == None:
            self.z1 = np.minimum.reduce(np.ravel(pix))
        if self.z2 == None:
            self.z2 = np.maximum.reduce(np.ravel(pix))

        # If the user has not selected a specific buffer for the display,
        # select and set the frame buffer size based on input image size.
        if bufname == 'iraf':
            useiraf = True
            bufname = None
        else:
            useiraf = False

        if bufname != None:
            _d.setFBconfig(None,bufname=bufname)
        else:
            _ny,_nx = pix.shape
            _d.selectFB(_nx,_ny,reset=1,useiraf=useiraf)

        # Initialize the specified frame buffer
        _d.setFrame(self.frame)
        _d.eraseFrame()

        # Apply user specified scaling to image, returns original
        # if none are specified.
        bpix = self._transformImage(pix)

        # Recompute the pixel range of (possibly) transformed array
        _z1 = self._transformImage(self.z1)
        _z2 = self._transformImage(self.z2)

        # If there was a problem in the transformation, then restore the original
        # array as the one to be displayed, even though it may not be ideal.
        if _z1 == _z2:
            if not quiet:
                log.warning('Error encountered during transformation. '
                            'No transformation applied...')
            bpix = pix
            self.z1 = np.minimum.reduce(np.ravel(bpix))
            self.z2 = np.maximum.reduce(np.ravel(bpix))
            # Failsafe in case input image is flat:
            if self.z1 == self.z2:
                self.z1 -= 1.
                self.z2 += 1.
        else:
            # Reset z1/z2 values now so that image gets displayed with
            # correct range.  Also, when displaying transformed images
            # this allows the input pixel value to be displayed, rather
            # than the transformed pixel value.
            self.z1 = _z1
            self.z2 = _z2

        _wcsinfo = nd.displaydev.ImageWCS(bpix,z1=self.z1,z2=self.z2,name=name)
        if not quiet:
            log.fullinfo('Image displayed with z1: {} z2: {}'.format(self.z1,
                                                                     self.z2))

        bpix = self._fbclipImage(bpix,_d.fbwidth,_d.fbheight)

        # Change pixel value to specified color if desired
        if masks is not None:
            if not isinstance(masks,list):
                masks = [masks]
            if mask_colors is None:
                # Set to red as default
                mask_colors = [204]*len(masks)
            for i in range(len(masks)):
                try:
                    if (masks[i][0].size>0 and masks[i][1].size>0):
                        bpix[masks[i]] = mask_colors[i]
                except TypeError:
                    pass

        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)

        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)

        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)
        #displaydev.close()


def propagate_gain(ext, gain_list, report_ext=False):
    """
    Propagate the gain into an output value when combining (mosaicking or
    tiling) multiple extensions. In addition, a warning is logged if the
    gains are not all the same *unless* the "display" function (primitive)
    is somewhere in the call stack. This is to prevent the warning appearing
    whenever raw or minimally-prepared data are displayed for quality
    assessment purposes.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension where the gain value is to be added
    gain_list: iterable
        input gain values from the extensions being combined
    report_ext: bool
        report the extension ID in the warning message?
    """
    log = logutils.get_logger(__name__)
    if not all(g == gain_list[0] for g in gain_list):
        if not any(rec.function == "display" for rec in inspect.stack()):
            id = f"{ext.filename} extension {ext.id}" if report_ext else f"{ext.filename}"
            log.warning(f"The extensions in {id} have different gains. "
                        "Run ADUToElectrons first?")
        with suppress(TypeError):  # some NoneTypes in gain
            ext.hdr[ext._keyword_for('gain')] = np.mean(gain_list)
