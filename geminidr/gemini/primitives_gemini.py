#
#                                                                  gemini_python
#
#                                                           primitives_gemini.py
# ------------------------------------------------------------------------------
import datetime
from contextlib import suppress
from copy import deepcopy

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.modeling import models

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at, transform

from geminidr.core import Bookkeeping, CalibDB, Preprocess
from geminidr.core import Visualize, Standardize, Stack, Stats

from .primitives_qa import QA
from . import parameters_gemini

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Gemini(Standardize, Bookkeeping, Preprocess, Visualize, Stack, QA,
             CalibDB, Stats):
    """
    This is the class containing the generic Gemini primitives.

    """
    tagset = {"GEMINI"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gemini)

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This primitive is used to add an Mask Definition File (MDF) extension to
        the input AstroData object. This MDF extension consists of a FITS binary
        table with information about where the spectroscopy slits are in
        the focal plane mask. In IFU, it is the position of the fibers. In
        Multi-Object Spectroscopy, it is the position of the multiple slits.
        In longslit is it the position of the single slit.

        If only one MDF is provided, that MDF will be added to all input AstroData
        object(s). If more than one MDF is provided, the number of MDF AstroData
        objects must match the number of input AstroData objects.

        If no MDF is provided, the primitive will attempt to determine an
        appropriate MDF.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mdf: str/None
            name of MDF to add (None => use default)
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        mdf_list = mdf or self.caldb.get_calibrations(adinputs, caltype="mask").files

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {ad.filename}, '
                            'since it has already been processed by addMDF')
                continue
            if hasattr(ad, 'MDF'):
                log.warning(f'An MDF extension already exists in {ad.filename},'
                            ' so no MDF will be added')
                continue

            if mdf is None:
                log.stdinfo(f'No MDF could be retrieved for {ad.filename}')
                continue

            try:
                # This will raise some sort of exception unless the MDF file
                # has a single MDF Table extension
                ad.MDF = mdf.MDF
            except:
                if len(mdf.tables) == 1:
                    ad.MDF = getattr(mdf, mdf.tables.pop())
                else:
                    log.warning(f'Cannot find MDF in {mdf.filename}, '
                                'so no MDF will be added')
                continue

            log.fullinfo(f'Attaching the MDF {mdf.filename} to {mdf.filename}')

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def standardizeObservatoryHeaders(self, adinputs=None, **params):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of Gemini data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        log.status("Updating keywords that are common to all Gemini data")
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, "
                            "since it has already been processed by "
                            "standardizeObservatoryHeaders")
                continue

            # Update various header keywords
            ad.hdr.set('BUNIT', 'adu', self.keyword_comments['BUNIT'])
            for ext in ad:
                if 'RADECSYS' in ext.hdr:
                    ext.hdr['RADESYS'] = (ext.hdr['RADECSYS'], ext.hdr.comments['RADECSYS'])
                    del ext.hdr['RADECSYS']

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            log.debug(f"Successfully updated keywords for {ad.filename}")
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        """
        This primitive is used to standardize the structure of Gemini data,
        specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        attach_mdf: bool
            attach an MDF to the AD objects?
        mdf: str
            full path of the MDF to attach
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # If attach_mdf=False, this just zips up the ADs with a list of Nones,
        # which has no side-effects.
        for ad, mdf in zip(*gt.make_lists(adinputs, params['mdf'])):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardizeStructure".
                            format(ad.filename))
                continue

            # Attach an MDF to each input AstroData object if it seems appropriate
            if params["attach_mdf"] and (ad.tags & {'LS', 'MOS', 'IFU', 'XD'}):
                # Some instruments don't accept an 'mdf' parameter because
                # they have a LUT instead of MDF files
                if 'mdf' in self.params['addMDF']:
                    self.addMDF([ad], mdf=mdf)
                else:
                    self.addMDF([ad])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def standardizeWCS(self, adinputs=None, **params):
        """
        This primitive attempts to identify inputs with a bad WCS based on the
        relationship between the WCS and other header keywords. If any such
        inputs are found, the reduction may either exit, or it may attempt to
        fix the WCS using header keywords describing the telescope offsets.
        In addition, it is also possible to construct entirely new WCS objects
        for each input based on the offsets.

        The primitive defines "groups" which are sequences of ADs with the
        same observation_id() and without a significant amount of dead time
        between successive exposures. Within each group, a base "Pointing" is
        constructed from the first AD with a self-consistent WCS, and the most
        recent self-consistent WCS is also stored. When an AD with a bad WCS is
        encountered, an attempt is made to fix it using the base Pointing first
        and then the most recent. This is done because only GNIRS can handle
        rotations between images and, if a sequence includes rotations, this
        rotation could become large over the course of a group, but might be
        small enough to ignore between successive images.

        Note that this method must be called BEFORE any instrument-specific
        WCS modifications, such as adding a spectroscopic axis. Child
        standardizeWCS() methods should super() this one before doing their
        own work.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        bad_wcs: str (exit | fix | new | ignore)
            how to handle a bad WCS, or whether to create a complete new set
        debug_consistency_limit: float
            maximum separation (in arcsec) between the WCS location and the
            expected location to not flag this AD object
        debug_max_deadtime: float
            maximum time (in seconds) between the end of the previous exposure
            and the start of this one for them to be considered part of the
            same group (and hence have the same base pointing position), if
            the observation_id()s also agree
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        bad_wcs = params["bad_wcs"]
        limit = params["debug_consistency_limit"]
        max_deadtime = params["debug_max_deadtime"]

        want_to_fix = bad_wcs not in ('check', 'exit', 'ignore')
        bad_wcs_list = []
        base_pointing = None
        last_pointing = None
        last_obsid = None
        last_endtime = None
        for ad in adinputs:
            if ad.tags.intersection({'ARC', 'BIAS', 'DARK', 'FLAT', 'PINHOLE'}):
                log.debug(f"Skipping {ad.filename} due to its tags")
                continue
            if (ad.instrument() == 'NIRI' and ad.is_ao() and
                    ad.phu.get('CRFOLLOW') == 'no'):
                log.fullinfo(f"Skipping {ad.filename} as the Cass rotator is fixed")
                continue

            try:
                start = ad.phu['UTSTART']
            except KeyError:
                this_starttime = ad.ut_datetime()
            else:
                if '.' not in start:
                    start + '.'
                if len(start) not in (12, 15):
                    start += '0' * (15 - len(start))
                this_starttime = datetime.datetime.combine(
                    ad.ut_date(), datetime.time.fromisoformat(start))

            this_obsid = ad.observation_id()
            if last_endtime is not None and (this_obsid != last_obsid or
                    (this_starttime - last_endtime).seconds > max_deadtime):
                if base_pointing is None:
                    raise ValueError(f"Now processing {ad.filename} as part "
                                     "of a new group, but could not find a "
                                     "valid pointing in the previous group")
                log.debug(f"Starting new group with {ad.filename}")
                base_pointing = None
                last_pointing = None

            try:
                p = Pointing(ad, log_function=self.log.debug)
            except ValueError:
                log.warning(f"{ad.filename} (and maybe other files) do not "
                            "have detector offsets. Cannot check/fix WCS.")
                # Ensure correct logging
                bad_wcs_list = []
                base_pointing = 1
                break

            selfcon = p.self_consistent(limit=limit)
            if not selfcon:
                log.stdinfo(f"{ad.filename}: WCS inconsistent with target "
                            "coordinates")
            needs_fixing = (bad_wcs == 'new' or not selfcon)
            if base_pointing is not None:
                needs_fixing |= not base_pointing.consistent_with(p)

            if needs_fixing:
                if bad_wcs == 'new' and base_pointing is None:
                    # Create a new base Pointing and update this AD
                    log.stdinfo(p.fix_pointing())
                    for ext, wcs in zip(ad, p.wcs):
                        ext.wcs = wcs
                    base_pointing = p
                    log.stdinfo(f"Using {ad.filename} as base pointing")
                elif not want_to_fix or base_pointing is None:
                    # Do not want to, or cannot yet, fix
                    bad_wcs_list.append(ad)
                else:
                    # Want to, and can, fix, so fix!
                    if last_pointing is None:
                        log.stdinfo(base_pointing.fix_wcs(ad))
                    else:  # can try both the base and the last Pointings
                        try:
                            log.stdinfo(base_pointing.fix_wcs(ad))
                        except NotImplementedError:
                            log.debug(f"Could not fix {ad.filename} using "
                                      f"{base_pointing.filename}")
                            log.stdinfo(last_pointing.fix_wcs(ad))

            if not needs_fixing:
                last_pointing = p
                if base_pointing is None:
                    # Found a reliable base WCS
                    base_pointing = p
                    log.stdinfo(f"Using {ad.filename} as base pointing")
                    # Fix all backed up ADs if we want to
                    if want_to_fix:
                        while bad_wcs_list:
                            log.stdinfo(base_pointing.fix_wcs(bad_wcs_list.pop(0)))

            # UTEND time is wrong for F2 data before 2015 Nov 30; it's the end
            # of the *previous* exposure. Hence just add the exposure time to
            # the start time to try to estimate the exposure end.
            if 'UTEND' not in ad.phu or (ad.instrument() == 'F2' and this_starttime <
                    datetime.datetime(year=2015, month=12, day=1)):
                last_endtime = this_starttime + datetime.timedelta(
                    seconds=ad.exposure_time())
            else:
                end = ad.phu['UTEND']
                if '.' not in end:
                    end += '.'
                if len(end) not in (12, 15):
                    end += '0' * (15 - len(end))
                last_endtime = datetime.datetime.combine(
                    ad.ut_date(), datetime.time.fromisoformat(end))
            last_obsid = this_obsid

        if not want_to_fix and bad_wcs_list:
            log.stdinfo("\nThe following files were identified as having bad"
                        " WCS information:")
            for ad in bad_wcs_list:
                log.stdinfo(f"    {ad.filename}")
            if base_pointing is None:
                log.stdinfo("\nNo valid base pointing was identified, so a "
                            "'new' WCS is required, 'fix' will not work.")
            if bad_wcs == 'exit':
                raise ValueError("Some files have bad WCS information and "
                                 "user has requested an exit")
            else:
                log.stdinfo("No changes are being made to the WCS information.")

        for ad in adinputs:
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def checkWCS(self, adinputs=None):
        """
        This primitive checks for consistency within the WCS by calling
        standardizeWCS().
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if len(set(ad.instrument() for ad in adinputs)) > 1:
            raise ValueError("Not all data are from the same instrument")
        if any('PREPARED' in ad.tags for ad in adinputs):
            raise ValueError(f"'{self.myself()}' requires unprepared data")

        # We don't want to modify the filename or PHU in case such changes
        # cause the outputs to be written to disk
        Gemini([]).standardizeWCS(adinputs, bad_wcs='ignore', suffix=None)
        for ad in adinputs:
            with suppress(KeyError):
                del ad.phu[self.timestamp_keys['standardizeWCS']]

        return adinputs

    def _add_longslit_wcs(self, ad, pointing=None):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        Parameters
        ----------
        ad: AstroData
            object to have a longslit WCS added
        pointing: None/tuple/"center"
            pointing center for reprojecting the WCS:
                None => don't reproject
                "center" => reproject to WCS at center of first extension

        """
        log = self.log
        # Need to exclude darks from having a spectroscopic WCS added as
        # they don't have a SPECT tag and will gum up the works. This only
        # needs to be done for F2's makeLampFlat as it uses flats minus
        # darks to remove dark current.
        if 'DARK' in ad.tags:
            log.stdinfo(f"{ad.filename} is a DARK, continuing")
            return ad

        log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
        if ad.dispersion() is None:
            raise ValueError(f"Unknown dispersion for {ad.filename}")
        cenwave = ad.actual_central_wavelength(asNanometers=True)
        if pointing == "center":
            pointing = ad[0].wcs(*(0.5 * np.array(ad[0].shape[::-1])))
        transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                   pointing=pointing)

        return ad

class Pointing:
    """
    A class that holds some information about the telescope pointing, both
    from the PHU keywords and the WCS keywords. The class needs to contain
    enough information to determine whether these two locations are consistent,
    and to create a new WCS from the PHU information.
    """
    # (x, y), 0-indexed
    center_of_rotation_dict = {'GNIRS': (629., 519.)}

    def __init__(self, ad, log_function=None):
        self.phu = ad.phu.copy()
        self.instrument = ad.instrument()
        self.pa = ad.position_angle()
        self.xoffset = ad.detector_x_offset()
        self.yoffset = ad.detector_y_offset()
        if self.xoffset is None or self.yoffset is None:
            raise ValueError("Cannot determine detector offsets")
        self.pixel_scale = ad.pixel_scale()
        self.filename = ad.filename
        self.wcs = [ext.wcs for ext in ad]
        self.logit = log_function or print

        self.logit(f"Pointing of {self.filename}")
        self.logit(f"X, Y offsets: ({self.xoffset:.3f}, {self.yoffset:.3f}) PA={self.pa}")
        self.target_coords = SkyCoord(ad.target_ra(), ad.target_dec(),
                                      unit=u.deg)
        # This returns the CRVALi keywords, which are expected to change
        # because the CRPIXj keywords are expected to be constant.
        self.coords = SkyCoord(ad.wcs_ra(), ad.wcs_dec(), unit=u.deg)

        # We want to know whether East is to the left or right when North
        # is up, which may depend on the port the instrument is on.
        # Unfortunately, we can't trust the CD matrix because values close
        # to zero (when the instrument is oriented along one of the cardinal
        # axes) can have random signs. So we need to do this empirically.
        self.flipped = (self.instrument == "GNIRS" or
                        self.phu.get("INPORT") == 1 and (
                                "GMOS" in self.instrument or
                                self.instrument == "GSAOI" or
                                self.instrument == "NIRI" and not ad.is_ao()))
        # If an offset is positive, it means that an object's pixel coords
        # will *decrease*, so the telescope has moved either up or right.
        self.expected_coords = self.target_coords.directional_offset_by(
            self.pa * u.deg, -self.yoffset * self.pixel_scale * u.arcsec).directional_offset_by(
            (self.pa + (90 if self.flipped else -90)) * u.deg, -self.xoffset * self.pixel_scale * u.arcsec
        )

    def __repr__(self):
        return f"Pointing object from {self.filename}"

    def self_consistent(self, limit=10):
        """
        Determine whether the WCS information in this Pointing is
        self-consistent and therefore (presumably) reliable. This
        is done by determining the sky distance between the WCS
        coordinates and the target+offset coordinates.

        Parameters
        ----------
        limit: float
            maximum discrepancy (in arcseconds) between the expected and
            actual pointings for the pointing to be considered OK

        Returns
        -------
        bool: is the pointing self-consistent?
        """
        sep = self.coords.separation(self.expected_coords).arcsec
        self.logit(f"Self-consistency report on {self.filename} "
                   f"(flipped={self.flipped})")
        self.logit(f"Base coords:     {self.coords.to_string(precision=5)}")
        self.logit("Expected coords: "
                   f"{self.expected_coords.to_string(precision=5)}")
        self.logit(f"Separation: {sep:.2f} arcsec")
        return sep <= limit

    def consistent_with(self, other):
        """
        Determine whether the WCS information in two Pointings is consistent.
        The detector offsets indicate how sources move, so if an offset is +20
        between two frames, then the sources will move 20 pixels in the +ve
        direction. The test is made by computing the pixel coordinates of the
        base Pointing's reference (RA, Dec) in both Pointings and confirming
        that the pixel distance between these is at least half what it expected
        from the detector offsets (if the expected distance is at least 10
        pixels). This is a scalar calculation, rather than checking each
        coordinate separately, in order to avoid issues where the expected
        difference is small.

        Parameters
        ----------
        other: Pointing object
            a second Pointing

        Returns
        -------
        bool: are these pointings consistent?
        """
        self.logit("Consistency report between base {} and {}".format(
            self.filename, other.filename))
        for wcs1, wcs2 in zip(self.wcs, other.wcs):
            try:
                ra, dec = at.get_center_of_projection(wcs1)
            except TypeError:  # if this returns None
                self.logit("Could not get center of projection!")
                return False
            x, y = am.get_named_submodel(wcs1.forward_transform, 'SKY').inverse(ra, dec)
            self.logit("Base: ({:.7f}, {:.7f}) -> ({:.3f}, {:.3f})".format(ra, dec, x, y))
            x2, y2 = am.get_named_submodel(wcs2.forward_transform, 'SKY').inverse(ra, dec)
            self.logit("New: ({:.3f}, {:.3f}) => relative offsets "
                       "({:.3f}, {:.3f})".format(x2, y2, x2-x, y2-y))
            dx = other.xoffset - self.xoffset
            dy = other.yoffset - self.yoffset
            self.logit("Relative header offsets: ({:.3f}, {:.3f})".format(dx, dy))
            distsq = dx * dx + dy * dy
            if distsq > 100 and (x-x2)**2 + (y-y2)**2 < 0.25 * distsq:
                self.logit("Scalar distance error!")
                return False
            if (abs(dx) > 2 and (x2-x) * dx < 0) or (abs(dy) > 2 and (y2-y) * dy < 0):
                self.logit("Sign error in one or both axes!")
                return False
        return True

    def fix_pointing(self):
        """
        Fix the WCS objects in this Pointing object based on the target
        coordinates and RA/DEC offsets. We do not change the pixel location
        of the projection center.

        Returns
        -------
        str: message indicating how the WCS has been fixed
        """
        for wcs in self.wcs:
            for m in wcs.forward_transform:
                if isinstance(m, models.AffineTransformation2D):
                    aftran = m
                elif isinstance(m, models.RotateNative2Celestial):
                    nat2cel = m
                    break
            else:
                raise ValueError("Cannot find center point of projection")
            # Update projection center with coordinates
            nat2cel.lon = self.expected_coords.ra.value
            nat2cel.lat = self.expected_coords.dec.value
            # Try to construct a new CD matrix.
            new_matrix = self.pixel_scale / 3600 * np.identity(2)
            pa = -self.pa if self.flipped else self.pa
            new_matrix = np.asarray(models.Rotation2D(pa)(*new_matrix))
            if not self.flipped:
                new_matrix[0] *= -1
            aftran.matrix = new_matrix

        return f"Reset WCS for {self.filename} using target coordinates"

    def fix_wcs(self, ad):
        """
        Fix another AD based on this pointing. The aim here is to preserve
        the pixel location around which the Pix2Sky projection occurs.

        Parameters
        ----------
        ad: AstroData object
            the AD whose WCS needs fixing

        Returns
        -------
        str: message indicating how the WCS has been fixed
        """
        xoffset, yoffset = ad.detector_x_offset(), ad.detector_y_offset()
        delta_pa = self.phu['PA'] - ad.phu['PA']
        rotate = abs(delta_pa) > 0.1
        if rotate:
            try:
                x0, y0 = self.center_of_rotation_dict[ad.instrument()]
            except KeyError:
                raise NotImplementedError("No center of rotation defined for "
                                          f"{ad.instrument()}. Please contact "
                                          "the HelpDesk for advice.")
            t = ((models.Shift(-xoffset - x0) & models.Shift(-yoffset - y0)) |
                 models.Rotation2D(delta_pa) |
                 (models.Shift(self.xoffset + x0) & models.Shift(self.yoffset + y0)))
        else:
            t = models.Shift(self.xoffset - xoffset) & models.Shift(self.yoffset - yoffset)

        # Copy the WCS of each extension of the "Pointing" AD, but update
        # the (RA, DEC) of the projection center, and the rotation matrix
        for ext, wcs in zip(ad, self.wcs):
            for m in wcs.forward_transform:
                if isinstance(m, models.AffineTransformation2D):
                    aftran = m
                elif isinstance(m, models.RotateNative2Celestial):
                    nat2cel = m
                    break
            else:
                raise ValueError("Cannot find center point of projection")
            x, y = wcs.invert(nat2cel.lon.value, nat2cel.lat.value)
            xnew, ynew = t(x, y)
            new_lon, new_lat = wcs(xnew, ynew)
            ext.wcs = deepcopy(wcs)
            for m in ext.wcs.forward_transform:
                if isinstance(m, models.AffineTransformation2D) and rotate:
                    m.matrix = models.Rotation2D(delta_pa)(*aftran.matrix.value)
                elif isinstance(m, models.RotateNative2Celestial):
                    m.lon = new_lon
                    m.lat = new_lat

        return f"Fixing the WCS for {ad.filename} using {self.filename}"
