#
#                                                                  gemini_python
#
#                                                       primitives_photometry.py
# ------------------------------------------------------------------------------
import numpy as np
from astropy.wcs import WCS
from astropy.stats import sigma_clip

from gempy.gemini import gemini_tools as gt
from gempy.gemini.gemini_catalog_client import get_fits_table
from gempy.gemini.eti.sextractoreti import SExtractorETI
from gempy.utils import logutils
from geminidr.gemini.lookups import color_corrections

from geminidr import PrimitivesBASE
from .parameters_photometry import ParametersPhotometry

from gempy.library.newmatch import match_catalogs, CallableWCS

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Photometry(PrimitivesBASE):
    """
    This is the class containing all of the primitives for photometry.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Photometry, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersPhotometry

    def addReferenceCatalog(self, adinputs=None, **params):
        """
        This primitive calls the gemini_catalog_client module to query a
        catalog server and construct a fits table containing the catalog data

        That module will query either gemini catalog servers or vizier.
        Currently, sdss9 and 2mass (point source catalogs are supported.

        For example, with sdss9, the FITS table has the following columns:

        - 'Id'       : Unique ID. Simple running number
        - 'Cat-id'   : SDSS catalog source name
        - 'RAJ2000'  : RA as J2000 decimal degrees
        - 'DEJ2000'  : Dec as J2000 decimal degrees
        - 'umag'     : SDSS u band magnitude
        - 'e_umag'   : SDSS u band magnitude error estimage
        - 'gmag'     : SDSS g band magnitude
        - 'e_gmag'   : SDSS g band magnitude error estimage
        - 'rmag'     : SDSS r band magnitude
        - 'e_rmag'   : SDSS r band magnitude error estimage
        - 'imag'     : SDSS i band magnitude
        - 'e_imag'   : SDSS i band magnitude error estimage
        - 'zmag'     : SDSS z band magnitude
        - 'e_zmag'   : SDSS z band magnitude error estimage

        With 2mass, the first 4 columns are the same, but the photometry
        columns reflect the J H and K bands.

        This primitive then adds the fits table catalog to the Astrodata
        object as 'REFCAT'

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        radius: float
            search radius (in degrees)
        source: str
            identifier for server to be used for catalog search
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        source = params["source"]
        radius = params["radius"]

        for ad in adinputs:
            try:
                ra = ad.wcs_ra()
                dec = ad.wcs_dec()
                if type(ra) is not float:
                    raise ValueError("wcs_ra descriptor did not return a float.")
                if type(ra) is not float:
                    raise ValueError("wcs_dec descriptor did not return a float.")
            except:
                if "qa" in self.context:
                    log.warning("No RA/Dec in header of {}; cannot find "
                                "reference sources".format(ad.filename))
                    continue
                else:
                    raise

            log.fullinfo("Querying {} for reference catalog".format(source))
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refcat = get_fits_table(source, ra, dec, radius)

            if refcat is None:
               log.stdinfo("No reference catalog sources found for {}".
                            format(ad.filename))
            else:
                log.stdinfo("Found {} reference catalog sources for {}".
                            format(len(refcat), ad.filename))
                ad.REFCAT = refcat

                # Match the object catalog against the reference catalog
                # Update the refid and refmag columns in the object catalog
                if any(hasattr(ext, 'OBJCAT') for ext in ad):
                    ad = _match_objcat_refcat(ad, self.context)
                else:
                    log.warning("No OBJCAT found; not matching OBJCAT to REFCAT")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

    def detectSources(self, adinputs=None, **params):
        """
        Find x,y positions of all the objects in the input image. Append 
        a FITS table extension with position information plus columns for
        standard objects to be updated with position from addReferenceCatalog
        (if any are found for the field).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mask: bool
            apply DQ plane as a mask before detection?
        set_saturation: bool
            set the saturation level of the data for SExtractor?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        set_saturation = params["set_saturation"]
        # Setting mask_bits=0 is the same as not replacing bad pixels
        mask_bits = params["replace_flags"] if params["mask"] else 0

        # Will raise an Exception if SExtractor is too old or missing
        SExtractorETI().check_version()

        adoutputs = []
        for ad in adinputs:
            # Get a seeing estimate from the header, if available
            seeing_estimate = ad.phu.get('MEANFWHM')

            # Get the appropriate SExtractor input files
            dqtype = 'no_dq' if any(ext.mask is None for ext in ad) else 'dq'
            sexpars = {'config': self.sx_dict[dqtype, 'sex'],
                      'PARAMETERS_NAME': self.sx_dict[dqtype, 'param'],
                      'FILTER_NAME': self.sx_dict[dqtype, 'conv'],
                      'STARNNW_NAME': self.sx_dict[dqtype, 'nnw']}

            for ext in ad:
                # saturation_level() descriptor always returns level in ADU,
                # so need to multiply by gain if image is not in ADU
                if set_saturation:
                    sat_level = ext.saturation_level()
                    if ext.hdr.get('BUNIT', 'adu').lower() != 'adu':
                        sat_level *= ext.gain()
                    sexpars.update({'SATUR_LEVEL': sat_level})

                # If we don't have a seeing estimate, try to get one
                if seeing_estimate is None:
                    log.debug("Running SExtractor to obtain seeing estimate")
                    sex_task = SExtractorETI([ext], sexpars,
                                    mask_dq_bits=mask_bits, getmask=True)
                    sex_task.run()
                    # An OBJCAT is *always* attached, even if no sources found
                    seeing_estimate = _estimate_seeing(ext.OBJCAT)

                # Re-run with seeing estimate (no point re-running if we
                # didn't get an estimate), and get a new estimate
                if seeing_estimate is not None:
                    log.debug("Running SExtractor with seeing estimate "
                              "{:.3f}".format(seeing_estimate))
                    sexpars.update({'SEEING_FWHM': '{:.3f}'.
                                   format(seeing_estimate)})
                    sex_task = SExtractorETI([ext], sexpars,
                                    mask_dq_bits=mask_bits, getmask=True)
                    sex_task.run()
                    # We don't want to replace an actual value with "None"
                    temp_seeing_estimate = _estimate_seeing(ext.OBJCAT)
                    if temp_seeing_estimate is not None:
                        seeing_estimate = temp_seeing_estimate

                # Although the OBJCAT has been added to the extension, it
                # needs to be massaged into the necessary format
                # We're deleting the OBJCAT first simply to suppress the
                # "replacing" message in gt.add_objcat, which would otherwise
                # be a bit confusing
                _cull_objcat(ext)
                objcat = ext.OBJCAT
                del ext.OBJCAT
                ad = gt.add_objcat(ad, extver=ext.hdr['EXTVER'], replace=False,
                                   table=objcat, sx_dict=self.sx_dict)
                log.stdinfo("Found {} sources in {}:{}".format(len(ext.OBJCAT),
                                            ad.filename, ext.hdr['EXTVER']))
                if len(ext.OBJCAT) == 0:
                    del ext.OBJCAT

            # Run some profiling code on the best sources to produce a
            # more IRAF-like FWHM number, adding two columns to the OBJCAT
            # (PROFILE_FWHM, PROFILE_EE50)
            ad = _profile_sources(ad, seeing_estimate)

            # Timestamp and update filename, and append to output list
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs

    def measureCCAndAstrometry(self, adinputs=None, **params):
        """
        This primitive does several things. For every input image with an
        OBJCAT, it will try to add a REFCAT. If successful, it will then
        measure the CC, determine the astrometric solution and, if requested,
        update the WCS in the image header.

        Parameters
        ----------
        update_wcs: bool
            Update the WCS in the header after OBJCAT-REFCAT matching?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        adoutputs = []
        for ad in adinputs:
            if any(hasattr(ext, 'OBJCAT') for ext in ad):
                ad = self.addReferenceCatalog([ad])[0]
            else:
                log.stdinfo("No OBJCAT found in input, so no comparison to "
                        "reference sources will be performed")
                adoutputs.append(ad)
                continue

            if hasattr(ad, 'REFCAT'):
                ad = self.measureCC([ad])[0]
                ad = self.determineAstrometricSolution([ad])[0]
            else:
                log.stdinfo("No reference sources found; no comparison "
                            "will be performed")
            adoutputs.append(ad)

        if params["correct_wcs"]:
            adoutputs = self.updateWCS(adoutputs)
        return adoutputs

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _match_objcat_refcat(ad, context='qa'):
    """
    Match the sources in the objcats against those in the corresponding
    refcat. Update the refid column in the objcat with the Id of the 
    catalog entry in the refcat. Update the refmag column in the objcat
    with the magnitude of the corresponding source in the refcat in the
    band that the image is taken in.

    CJS: Since this is only called here, and only on single AD, its ability
    to handle inputs lists has been removed. If it gets used elsewhere, it
    should be moved to gemini_tools and make use of @accept_single_adinput

    Parameters
    ----------
    adinput: AstroData
        input image for OBJCAT-REFCAT matching
    """
    log = logutils.get_logger(__name__)

    filter_name = ad.filter_name(pretty=True)
    colterm_dict = color_corrections.colorTerms
    if filter_name in colterm_dict:
        formulae = colterm_dict[filter_name]
    else:
        log.warning("Filter {} is not in catalogs - will not be able to flux "
            "calibrate".format(filter_name))
        formulae = []

    # If there are no refcats, don't try to go through them.
    try:
        refcat = ad.REFCAT
    except AttributeError:
        log.warning("No REFCAT present - cannot match to OBJCAT")
        return ad
    if not any(hasattr(ext, 'OBJCAT') for ext in ad):
        log.warning("No OBJCATs in {} - cannot match to REFCAT".
                    format(ad.filename))
        return ad

    # Try to be clever here, and work on the extension with the highest
    # number of matches first, as this will give the most reliable offsets.
    # which can then be used to constrain the other extensions. The problem
    # is we don't know how many matches we'll get until we do it, and that's
    # slow, so use OBJCAT length as a proxy.
    objcat_lengths = [len(ext.OBJCAT) if hasattr(ext, 'OBJCAT') else 0
                      for ext in ad]
    objcat_order = np.argsort(objcat_lengths)[::-1]

    pixscale = ad.pixel_scale()
    initial = (15.0 if ad.instrument()=='GNIRS' else 5.0)/pixscale  # Search box size
    final = 1.0/pixscale     # Matching radius
    max_ref_sources = 100 if 'qa' in context else None  # Don't need more than this many

    working_model = (0, None)

    for index in objcat_order:
        extver = ad[index].hdr['EXTVER']
        try:
            objcat = ad[index].OBJCAT
        except AttributeError:
            log.stdinfo('No OBJCAT in {}:{}'.format(ad.filename, extver))
            continue
        objcat_len = len(objcat)

        # The coordinates of the reference sources are corrected
        # to pixel positions using the WCS of the object frame
        wcs = WCS(ad.header[index+1])
        xref, yref = wcs.all_world2pix(refcat['RAJ2000'],
                                       refcat['DEJ2000'], 1)

        # Reduce the search space if we've previously found a match
        # TODO: This code is more generic than it needs to be now (the model
        # only has unfixed offsets) but less generic than it will need to be
        # if a rotation or magnification is added)
        m_init = working_model[1]
        if m_init:
            initial = 2.5 / pixscale
            for param in [getattr(m_init, p) for p in m_init.param_names]:
                if not param.fixed:
                    param.bounds = (param.value-initial, param.value+initial)

        # First: estimate number of reference sources in field
        # Inverse map ref coords->image plane and see how many are in field
        xx, yy = m_init.inverse(xref, yref) if m_init else (xref, yref)
        x1, y1 = 0, 0
        y2, x2 = ad[index].data.shape
        # Could tweak y1, y2 here for GNIRS
        in_field = np.all((xx>x1-initial, xx<x2+initial, yy>y1-initial,
                           yy<y2+initial), axis=0)
        num_ref_sources = np.sum(in_field)

        # We probably don't need zillions of REFCAT sources
        if max_ref_sources and num_ref_sources > max_ref_sources:
            # imag and kmag are typically the deepest catalogues
            if 'imag' in refcat.colnames:
                magcol = 'imag'
            elif 'kmag' in refcat.colnames:
                magcol = 'kmag'
            else:
                try:
                    magcol = [c for c in refcat.colnames if 'mag' in c.lower()
                              and not 'image' in c.lower()  # "mag" in "image"!
                              and not 'err' in c .lower()][0]
                except:
                    magcol = None
                    log.stdinfo('Cannot find a magnitude column to cull REFCAT')
            if magcol:
                sorted_args = np.argsort(refcat[magcol])
                in_field = sorted_args[in_field[sorted_args]][:max_ref_sources]
                log.stdinfo('Culling REFCAT based on {} for speed'.format(magcol))
                # in_field is now a list of indices, not a boolean array
                num_ref_sources = len(in_field)

        # How many objects do we want to try to match? Keep brightest ones only
        if objcat_len > 2*num_ref_sources:
            keep_num = max(2*num_ref_sources, min(10,objcat_len))
        else:
            keep_num = objcat_len
        sorted_idx = np.argsort(objcat['MAG_AUTO'])[:keep_num]

        m = CallableWCS(wcs, direction=-1)
        m.x_offset.bounds = (-initial, initial)
        m.y_offset.bounds = (-initial, initial)
        m.factor.fixed = True
        m.angle.fixed = True
        #working_model = (0, m)

        if num_ref_sources > 0:
            log.stdinfo('Matching extver {} with {} REFCAT and {} OBJCAT sources'.
                        format(extver, num_ref_sources, keep_num))
            matched, m_final = match_catalogs(objcat['X_IMAGE'], objcat['Y_IMAGE'],
                                              xref, yref, use_in=sorted_idx, use_ref=in_field,
                                              model_guess=working_model[1], translation_range=initial,
                                              tolerance=0.1, match_radius=final)
            #xref, yref = ad.REFCAT['RAJ2000'], ad.REFCAT['DEJ2000']
            #matched, m_final = match_catalogs(xref, yref, objcat['X_IMAGE'], objcat['Y_IMAGE'],
            #                                  use_in=in_field, use_ref=sorted_idx,
            #                                  model_guess=working_model[1], translation_range=initial,
            #                                  tolerance=0.1, match_radius=final)
        else:
            log.stdinfo('No REFCAT sources in field of extver {}'.format(extver))
            continue

        num_matched = sum(m>0 for m in matched)
        log.stdinfo("Matched {} objects in OBJCAT:{} against REFCAT".
                    format(num_matched, extver))
        # If this is a "better" match, save it
        # TODO? Some sort of averaging of models?
        if num_matched > max(working_model[0], 2):
            working_model = (num_matched, m_final)

        # Loop through the reference list updating the refid in the objcat
        # and the refmag, if we can
        for i, m in enumerate(matched):
            if m >= 0:
                objcat['REF_NUMBER'][i] = refcat['Id'][m]
                # Assign the magnitude
                if formulae:
                    mag, mag_err = _calculate_magnitude(formulae, refcat, m)
                    objcat['REF_MAG'][i] = mag
                    objcat['REF_MAG_ERR'][i] = mag_err
    return ad

def _calculate_magnitude(formulae, refcat, indx):
    # This is a bit ugly: we want to iterate over formulae so we must
    # nest a single formula into a list
    if not isinstance(formulae[0], list):
        formulae = [formulae]

    mags = []
    mag_errs = []
    for formula in formulae:
        mag = 0.0
        mag_err_sq = 0.0
        for term in formula:
            # single filter
            if type(term) is str:
                if term+'mag' in refcat.columns:
                    mag += refcat[term+'mag'][indx]
                    mag_err_sq += refcat[term+'mag_err'][indx]**2
                else:
                    # Will ensure this magnitude is not used
                    mag = np.nan
            # constant (with uncertainty)
            elif len(term) == 2:
                mag += float(term[0])
                mag_err_sq += float(term[1])**2
            # color term (factor, uncertainty, color)
            elif len(term) == 3:
                filters = term[2].split('-')
                if len(filters)==2 and np.all([f+'mag' in refcat.columns
                                               for f in filters]):
                    col = refcat[filters[0]+'mag'][indx] - \
                        refcat[filters[1]+'mag'][indx]
                    mag += float(term[0])*col
                    dmagsq = refcat[filters[0]+'mag_err'][indx]**2 + \
                        refcat[filters[1]+'mag_err'][indx]**2
                    # When adding a (H-K) color term, often H is a 95% upper limit
                    # If so, we can only return an upper limit, but we need to
                    # account for the uncertainty in K-band 
                    if np.isnan(dmagsq):
                        mag -= 1.645*np.sqrt(mag_err_sq)
                    mag_err_sq += ((term[1]/term[0])**2 + dmagsq/col**2) * \
                        (float(term[0])*col)**2
                else:
                    mag = np.nan        # Only consider this if values are sensible
        if not np.isnan(mag):
            mags.append(mag)
            mag_errs.append(np.sqrt(mag_err_sq))
    
    # Take the value with the smallest uncertainty (NaN = large uncertainty)
    if mags:
        lowest = np.argmin(np.where(np.isnan(mag_errs),999,mag_errs))
        return mags[lowest], mag_errs[lowest]
    else:
        return -999, -999

def _estimate_seeing(objcat):
    """
    This function tries to estimate the seeing from a SExtractor object
    catalog, so future runs of SExtractor can provide better CLASS_STAR
    classifications. This uses a catalog that hasn't yet been run through
    _profile_sources() so lacks the extra columns that
    gemini_tools.clip_sources() needs.

    Parameters
    ----------
    objcat: an OBJCAT instance

    Returns
    -------
    float: the seeing estimate (or None)
    """
    try:
        badpix = objcat['NIMAFLAGS_ISO']
    except KeyError:
        badpix = np.zeros_like(objcat['NUMBER'])

    # Convert FWHM_WORLD from degrees to arcseconds
    objcat['FWHM_WORLD'] *= 3600

    # Only use objects that are: fairly round
    #                            thought to be stars by SExtractor
    #                            decent S/N ratio
    #                            unflagged (blended, saturated is OK)
    #                            not many bad pixels
    good = np.logical_and.reduce([objcat['ISOAREA_IMAGE'] > 20,
                                  objcat['B_IMAGE'] > 1.1,
                                  objcat['ELLIPTICITY'] < 0.5,
                                  objcat['CLASS_STAR'] > 0.8,
                                  objcat['FLUX_AUTO'] > 25*objcat['FLUXERR_AUTO'],
                                  objcat['FLAGS'] & 65528 == 0,
                                  objcat['FWHM_WORLD'] > 0,
                                  badpix < 0.2*objcat['ISOAREA_IMAGE']])
    good_fwhm = objcat['FWHM_WORLD'][good]
    if len(good_fwhm) > 3:
        seeing_estimate = sigma_clip(good_fwhm, sigma=3, iters=1).mean()
    elif len(good_fwhm) > 0:
        seeing_estimate = np.mean(good_fwhm)
    else:
        seeing_estimate = None

    if seeing_estimate <= 0:
        seeing_estimate = None

    return seeing_estimate

def _cull_objcat(ext):
    """
    Takes an extension of an AD object with attached OBJCAT (and possibly
    OBJMASK) and culls the OBJCAT of crap. If the OBJMASK exists, it also
    edits that to remove pixels associated with these sources. Finally, it
    renumbers the 'NUMBER' column into a contiguous sequence.

    Parameters
    ----------
    ext: a single extension of an AD object
    """
    try:
        objcat = ext.OBJCAT
    except AttributeError:
        return ext

    all_objects = objcat['NUMBER']
    # Remove sources of less than 20 pixels
    objcat.remove_rows(objcat['ISOAREA_IMAGE'] < 20)
    # Remove implausibly narrow sources
    objcat.remove_rows(objcat['B_IMAGE'] < 1.1)
    # Remove *really* bad sources. "Bad" pixels might be saturated, but the
    # source is still real, so be very conservative
    if 'NIMAFLAGS_ISO' in objcat.columns:
        objcat.remove_rows(objcat['NIMAFLAGS_ISO'] > 0.9*objcat['ISOAREA_IMAGE'])

    # Create new OBJMASK with 1 only for unculled objects
    if hasattr(ext, 'OBJMASK'):
        objmask_shape = ext.OBJMASK.shape
        ext.OBJMASK = np.where(np.in1d(ext.OBJMASK.ravel(), objcat['NUMBER']),
                               1, 0).reshape(objmask_shape).astype(np.uint8)

    # Now renumber what's left sequentially
    objcat['NUMBER'].data[:] = range(1, len(objcat)+1)
    return ext

def _profile_sources(ad, seeing_estimate=None):
    """
    FWHM (and encircled-energy) measurements of objects to be more IRAF-like.
    Finds the distance from the source center to the closest pixel whose flux
    is less than half the peak. Also finds the distance to the farthest pixel
    whose flux is more than half the peak, provided this is closer than the
    10th closest pixel below half the peak (the number 10 is arbitrary, but
    ensures that it's finding a pixel that's genuinely part of the profile,
    and not some cosmic ray or nearby source. These radii are averaged to
    give the HWHM, which is doubled to give the FWHM.
    
    The 50% encircled energy (EE50) is just determined from a cumulative sum
    of pixel values, sorted by distance from source center. 
    """
    for ext in ad:
        try:
            objcat = ext.OBJCAT
        except AttributeError:
            continue

        catx = objcat["X_IMAGE"]
        caty = objcat["Y_IMAGE"]
        catbg = objcat["BACKGROUND"]
        cattotalflux = objcat["FLUX_AUTO"]
        catmaxflux = objcat["FLUX_MAX"]
        data = ext.data
        if seeing_estimate is None:
            stamp_size = max(10,int(0.5/ext.pixel_scale()))
        else:
            stamp_size = max(10,int(1.2*seeing_estimate/ext.pixel_scale()))
        # Make a default grid to use for distance measurements
        dist = np.mgrid[-stamp_size:stamp_size,-stamp_size:stamp_size]+0.5

        fwhm_list = []
        e50d_list = []
        newmax_list = []
        for i in range(0, len(objcat)):
            xc = catx[i] - 0.5
            yc = caty[i] - 0.5
            bg = catbg[i]
            tf = cattotalflux[i]
            mf = catmaxflux[i]

            # Check that there's enough room for a stamp
            sz = stamp_size
            if (int(yc)-sz<0 or int(xc)-sz<0 or
                int(yc)+sz>=data.shape[0] or int(xc)+sz>=data.shape[1]):
                fwhm_list.append(-999)
                e50d_list.append(-999)
                newmax_list.append(mf)
                continue

            # Estimate new FLUX_MAX from pixels around peak
            mf = np.max(data[int(yc)-2:int(yc)+3,int(xc)-2:int(xc)+3]) - bg
            # Bright sources in IR images can "volcano", so revert to
            # catalog value if these pixels are negative
            if mf < 0:
                mf = catmaxflux[i]

            # Get image stamp around center point
            stamp=data[int(yc)-sz:int(yc)+sz,int(xc)-sz:int(xc)+sz]

            # Reset grid to correct center coordinates
            shift_dist = dist.copy()
            shift_dist[0] += int(yc)-yc
            shift_dist[1] += int(xc)-xc
    
            # Square root of the sum of the squares of the distances
            rdistsq = np.sum(shift_dist**2,axis=0)

            # Radius and flux arrays for the radial profile
            rpr = rdistsq.flatten()
            rpv = stamp.flatten() - bg
    
            # Sort by the radius
            sort_order = np.argsort(rpr) 
            radsq = rpr[sort_order]
            flux = rpv[sort_order]

            # Count pixels above half flux and circularize this area
            # Do one iteration in case there's a neighbouring object
            hwhmsq = np.sum(flux>halfflux)/np.pi
            hwhm = np.sqrt(np.sum(flux[radsq<1.5*hwhmsq]>halfflux)/np.pi)
            if hwhm < stamp_size:
                fwhm_list.append(2*hwhm)
            else:
                fwhm_list.append(-999)

            # Find the first radius that encircles half the total flux
            sumflux = np.cumsum(flux)
            halfflux = 0.5 * tf
            first_50pflux = np.where(sumflux>=halfflux)[0]
            if first_50pflux.size>0:
                e50d_list.append(2*np.sqrt(radsq[first_50pflux[0]]))
            else:
                e50d_list.append(-999)

            newmax_list.append(mf)

        objcat["PROFILE_FWHM"][:] = np.array(fwhm_list)
        objcat["PROFILE_EE50"][:] = np.array(e50d_list)
        objcat["FLUX_MAX"][:] = np.array(newmax_list)
    return ad
