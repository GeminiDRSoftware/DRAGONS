"""
This module holds the CalibrationGHOST class
"""
from gemini_obs_db.orm.header import Header
from gemini_obs_db.orm.ghost import Ghost
from gemini_calmgr.cal.calibration import Calibration, not_processed, not_imaging, not_spectroscopy

import math


class CalibrationGHOST(Calibration):
    """
    This class implements a calibration manager for GHOST.
    It is a subclass of Calibration
    """
    instrClass = Ghost
    instrDescriptors = (
        'disperser',
        'filter_name',
        'focal_plane_mask',
        'detector_x_bin',
        'detector_y_bin',
        'amp_read_area',
        'read_speed_setting',
        'gain_setting',
        'res_mode',
        'prepared',
        'overscan_trimmed',
        'overscan_subtracted'
        'want_before_arc'
        )

    def __init__(self, session, *args, **kwargs):
        # Need to super the parent class __init__ to get want_before_arc
        # keyword in
        super(CalibrationGHOST, self).__init__(session, *args, **kwargs)

        if self.descriptors is None and self.instrClass is not None:
            self.descriptors['want_before_arc'] = self.header.want_before_arc
            iC = self.instrClass
            query = session.query(iC).filter(
                iC.header_id == self.descriptors['header_id'])
            inst = query.first()

            # Populate the descriptors dictionary for the instrument
            for descr in self.instrDescriptors:
                self.descriptors[descr] = getattr(inst, descr, None)

        # Set the list of applicable calibrations
        self.set_applicable()

    def set_applicable(self):
        """
        This method determines which calibration types are applicable
        to the target data set, and records the list of applicable
        calibration types in the class applicable variable.
        All this really does is determine whether what calibrations the
        /calibrations feature will look for. Just because a caltype isn't
        applicable doesn't mean you can't ask the calmgr for one.
        """
        self.applicable = []

        if self.descriptors:

            # MASK files do not require anything,
            if self.descriptors['observation_type'] == 'MASK':
                return

            # PROCESSED_SCIENCE files do not require anything
            if 'PROCESSED_SCIENCE' in self.types:
                return

            # Do BIAS. Most things require Biases.
            require_bias = True

            if self.descriptors['observation_type'] in ('BIAS', 'ARC'):
                # BIASes and ARCs do not require a bias.
                require_bias = False

            elif self.descriptors['observation_class'] in ('acq', 'acqCal'):
                # acq images don't require a BIAS.
                require_bias = False

            elif self.descriptors['detector_roi_setting'] == 'Central Stamp':
                # Anything that's ROI = Central Stamp does not require a bias
                require_bias = False

            if require_bias:
                self.applicable.append('bias')
                self.applicable.append('processed_bias')

            # If it (is spectroscopy) and
            # (is an OBJECT) and
            # (is not a Twilight) and
            # (is not a specphot)
            # then it needs an arc, flat, spectwilight, specphot
            if ((self.descriptors['spectroscopy'] == True) and
                    (self.descriptors['observation_type'] == 'OBJECT') and
                    (self.descriptors['object'] != 'Twilight') and
                    (self.descriptors['observation_class'] not in ['partnerCal', 'progCal'])):
                self.applicable.append('arc')
                self.applicable.append('processed_arc')
                self.applicable.append('flat')
                self.applicable.append('processed_flat')
                self.applicable.append('spectwilight')
                self.applicable.append('specphot')


            # If it (is imaging) and
            # (is Imaging focal plane mask) and
            # (is an OBJECT) and (is not a Twilight) and
            # is not acq or acqcal
            # then it needs flats, processed_fringe
            if ((self.descriptors['spectroscopy'] == False) and
                     (self.descriptors['focal_plane_mask'] == 'Imaging') and
                     (self.descriptors['observation_type'] == 'OBJECT') and
                     (self.descriptors['object'] != 'Twilight') and
                     (self.descriptors['observation_class'] not in ['acq', 'acqCal'])):

                self.applicable.append('flat')
                self.applicable.append('processed_flat')
                self.applicable.append('processed_fringe')
                # If it's all that and obsclass science, then it needs a photstd
                # need to take care that phot_stds don't require phot_stds for recursion
                if self.descriptors['observation_class'] == 'science':
                    self.applicable.append('photometric_standard')

            # If it is MOS then it needs a MASK
            if 'MOS' in self.types:
                self.applicable.append('mask')

    @not_imaging
    def arc(self, processed=False, howmany=2, return_query=False):
        """
        This method identifies the best GHOST ARC to use for the target
        dataset.

        This will find GHOST arcs with matching wavelength within 0.001 microns, disperser, and filter name.
        If "want_before_arc" is set and true, it limits to 1 result and only matches observations prior to the
        ut_datetime.  If it is set and false, it limits to 1 result after the ut_datetime.  Otherwise, it keeps
        the `howmany` as specified with a default of 2 and has no restriction on ut_datetime.
        It matches within 1 year.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw arcs
        howmany : int, default 2 if `want_before_arc` is not set, or 1 if it is
            How many matches to return

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        ab = self.descriptors.get('want_before_arc', None)
        # Default 2 arcs, hopefully one before and one after
        if ab is not None:
            howmany = 1
        else:
            howmany = howmany if howmany else 2
        filters = []
        # # Must match focal_plane_mask only if it's not the 5.0arcsec slit in the target, otherwise any longslit is OK
        # if self.descriptors['focal_plane_mask'] != '5.0arcsec':
        #     filters.append(Ghost.focal_plane_mask == self.descriptors['focal_plane_mask'])
        # else:
        #     filters.append(Ghost.focal_plane_mask.like('%arcsec'))

        if ab:
            # Add the 'before' filter
            filters.append(Header.ut_datetime < self.descriptors['ut_datetime'])
        elif ab is None:
            # No action required
            pass
        else:
            # Add the after filter
            filters.append(Header.ut_datetime > self.descriptors['ut_datetime'])

        # # The science amp_read_area must be equal or substring of the cal amp_read_area
        # # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # # have a subset of the amps thus we must do the substring match
        # if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
        #     filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        # elif self.descriptors['amp_read_area'] is not None:
        #         filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        query = (
            self.get_query()
                .arc(processed)
                .add_filters(*filters)
                .match_descriptors(Header.instrument,
                                   Header.camera,
                                   Ghost.disperser,
                                   Ghost.filter_name,
                                   Ghost.res_mode)
                .tolerance(central_wavelength=0.001)
                # Absolute time separation must be within 1 year
                .max_interval(days=365)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    def dark(self, processed=False, howmany=None, return_query=False):
        """
        Method to find best GHOST Dark frame for the target dataset.

        This will find GHOST darks with matching read speed setting, gain setting, and within 50 seconds
        exposure time.  It will also matching amp read area.  It matches within 1 year.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw darks
        howmany : int, default 1 if processed, else 15

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        if howmany is None:
            howmany = 1 if processed else 15

        filters = []
        # The science amp_read_area must be equal or substring of the cal amp_read_area
        # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # have a subset of the amps thus we must do the substring match
        if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
            filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        elif self.descriptors['amp_read_area'] is not None:
                filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        # Must match exposure time.


        query = (
            self.get_query()
                .dark(processed)
                .add_filters(*filters)
                .match_descriptors(Header.instrument,
                                   Ghost.read_speed_setting,
                                   Ghost.gain_setting)
                .tolerance(exposure_time = 50.0)
                # Absolute time separation must be within 1 year
                .max_interval(days=365)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    def bias(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the best bias frames for the target dataset

        This will find GHOST biases with matching read speed setting, gain setting, amp read area, and x and y binning.
        If it's 'prepared' data, it will match overscan trimmed and overscan subtracted.

        It matches within 90 days

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw biases
        howmany : int, default 1 if processed, else 50

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        if howmany is None:
            howmany = 1 if processed else 50

        filters = []
        # The science amp_read_area must be equal or substring of the cal amp_read_area
        # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # have a subset of the amps thus we must do the substring match
        if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
            filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        elif self.descriptors['amp_read_area'] is not None:
            filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        # The Overscan section handling: this only applies to processed biases
        # as raw biases will never be overscan trimmed or subtracted, and if they're
        # processing their own biases, they ought to know what they want to do.
        if processed:
            if self.descriptors['prepared'] == True:
                # If the target frame is prepared, then we match the overscan state.
                filters.append(Ghost.overscan_trimmed == self.descriptors['overscan_trimmed'])
                filters.append(Ghost.overscan_subtracted == self.descriptors['overscan_subtracted'])
            else:
                # If the target frame is not prepared, then we don't know what thier procesing intentions are.
                # we could go with the default (which is trimmed and subtracted).
                # But actually it's better to just send them what we have, as we has a mishmash of both historically
                #filters.append(Ghost.overscan_trimmed == True)
                #filters.append(Ghost.overscan_subtracted == True)
                pass

        query = (
            self.get_query()
                .bias(processed)
                .add_filters(*filters)
                .match_descriptors(Header.instrument,
                                   Header.camera,
                                  Ghost.detector_x_bin,
                                  Ghost.detector_y_bin,
                                  Ghost.read_speed_setting,
                                  Ghost.gain_setting)
                # Absolute time separation must be within 3 months
                .max_interval(days=90)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    def imaging_flat(self, processed, howmany, flat_descr, filt, sf=False, return_query=False):
        """
        Method to find the best imaging flats for the target dataset

        This will find imaging flats that are either obervation type of 'FLAT' or
        are both dayCal and 'Twilight'.  This also adds a large set of flat filters
        in flat_descr from the higher level flat query.

        This will find GHOST imaging flats with matching read speed setting, gain setting, filter name,
        res mode, focal plane mask, and disperser.

        It matches within 180 days

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw imaging flats
        howmany : int, default 1 if processed, else 20
            How many do we want results
        flat_descr: list
            set of filter parameters from the higher level function calling into this helper method
        filt: list
            Additional filter terms to apply from the higher level method
        sf: bool
            True for slit flats, else False

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        if howmany is None:
            howmany = 1 if processed else 20

        if processed:
            query = self.get_query().PROCESSED_FLAT()
        elif sf:
            # Find the relevant slit flat
            query = self.get_query().spectroscopy(
                False).observation_type('FLAT')
        else:
            # Imaging flats are twilight flats
            # Twilight flats are dayCal OBJECT frames with target Twilight
            query = self.get_query().raw().dayCal().OBJECT().object('Twilight')
        query = (
            query.add_filters(*filt)
                 .match_descriptors(*flat_descr)
                 # Absolute time separation must be within 6 months
                 .max_interval(days=180)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    def spectroscopy_flat(self, processed, howmany, flat_descr, filt, return_query=False):
        """
        Method to find the best imaging flats for the target dataset

        This will find spectroscopy flats with a central wavelength within 0.001 microns, a matching elevation, and
        matching cass rotator pa (for elevation under 85).  The specific tolerances for elevation
        depend on factors such as the type of focal plane mask.  The search also adds a large set of flat filters
        in flat_descr and filt from the higher level flat query.

        It matches within 180 days

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw imaging flats
        howmany : int, default 1 if processed, else 2
            How many do we want results
        flat_descr: list
            set of filter parameters from the higher level function calling into this helper method
        filt: list
            Additional filter terms to apply from the higher level method

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        if howmany is None:
            howmany = 1

        query = (
            self.get_query()
                .flat(processed)
                .add_filters(*filt)
                .match_descriptors(*flat_descr)

            # Absolute time separation must be within 6 months
                .max_interval(days=180)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    def flat(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the best GHOST FLAT fields for the target dataset

        This will find GHOST flats with matching read speed setting, gain setting, filter name,
        res mode, focal plane mask, and disperser.  It will search for matching spectroscopy setting
        and matching amp read area.  Then additional filtering is done based on logic either for
        imaging flats or spectroscopy flats, as per :meth:`spectroscopy_flat` and :meth:`imaging_flat`.

        It matches within 180 days

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw imaging flats
        howmany : int
            How many do we want

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        filters = []

        # Common descriptors for both types of flat
        # Must totally match instrument, detector_x_bin, detector_y_bin, filter
        flat_descriptors = (
            Header.instrument,
            Header.camera,
            Ghost.filter_name,
            Ghost.read_speed_setting,
            Ghost.gain_setting,
            Ghost.res_mode,
            Header.spectroscopy,
            # Focal plane mask must match for imaging too... To avoid daytime thru-MOS mask imaging "flats"
            Ghost.focal_plane_mask,
            Ghost.disperser, # this can be common-mode as imaging is always 'MIRROR'
            )

        # Not needed for GMOS?!
        # # The science amp_read_area must be equal or substring of the cal amp_read_area
        # # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # # have a subset of the amps thus we must do the substring match
        # if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
        #     flat_descriptors = flat_descriptors + (Ghost.amp_read_area,)
        # elif self.descriptors['amp_read_area'] is not None:
        #     filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        if self.descriptors['spectroscopy']:
            return self.spectroscopy_flat(processed, howmany, flat_descriptors, filters, return_query=return_query)
        else:
            return self.imaging_flat(processed, howmany, flat_descriptors, filters, return_query=return_query)

    def processed_slitflat(self, howmany=None, return_query=False):
        """
        Method to find the best GHOST SLITFLAT for the target dataset

        If the type is 'SLITV', this method falls back to the regular :meth:`flat` logic.

        This will find GHOST imaging flats with matching res mode.
        It filters further on the logic in :meth:`imaging_flat`.

        It matches within 180 days

        Parameters
        ----------

        howmany : int, default 1
            How many do we want results
        flat_descr: list
            set of filter parameters from the higher level function calling into this helper method
        filt: list
            Additional filter terms to apply from the higher level method
        sf: bool
            True for slit flats, else False

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # I think we *always* only want 1 slitflat, even if unprocessed, since
        # that means a bundle which has many individual exposures
        if howmany is None:
            howmany = 1

        if 'SLITV' in self.types:
            return self.flat(True, howmany, return_query=return_query)

        filters = []
        filters.append(Header.spectroscopy == False)

        # Here we're asking for a slitflat for an arm, so the detector modes relate to
        # different detectors. But the res_mode needs to match because different fibers
        # are illuminated in different res modes.
        flat_descriptors = (
            Header.instrument,
            Ghost.res_mode,
            )

        return self.imaging_flat(False, howmany, flat_descriptors, filters,
                                 sf=True, return_query=return_query)

    def processed_slit(self, howmany=None, return_query=False):
        """
        Method to find the best processed GHOST SLIT for the target dataset

        This will find GHOST processed slits.  It matches the observation
        type, res mode, and within 30 seconds.  For 'ARC' observation type it matches
        'PROCESSED_UNKNOWN' data, otherwise it matches 'PREPARED' data.

        Parameters
        ----------

        howmany : int
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        descripts = [
            Header.instrument,
            Header.observation_type,
            Ghost.res_mode,
            ]

        # We need to match exposure time for on-sky observations
        # (the exposure time has been munged in the processed_slit to match
        # the science exposure that needs it)
        if self.descriptors['observation_type'] not in ('ARC', 'BIAS', 'FLAT'):
            descripts.append(Header.exposure_time)

        query = (
            self.get_query()
                .reduction(  # this may change pending feedback from Kathleen
                    'PROCESSED_ARC' if
                    self.descriptors['observation_type'] == 'ARC' else
                    'PROCESSED_UNKNOWN'
                )
                .spectroscopy(False)
                .match_descriptors(*descripts)
                # Need to use the slit image that matches the input observation;
                # needs to match within 30 seconds!
                .max_interval(seconds=30)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)


    def processed_fringe(self, howmany=None, return_query=False):
        """
        Method to find the best GHOST processed fringe for the target dataset

        This will find GHOST processed fringes matching the amp read area, filter name, and x and y binning.
        It matches within 1 year.

        Parameters
        ----------

        howmany : int, default 1
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # Default number to associate
        howmany = howmany if howmany else 1

        filters = []
        # The science amp_read_area must be equal or substring of the cal amp_read_area
        # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # have a subset of the amps thus we must do the substring match
        if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
            filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        elif self.descriptors['amp_read_area'] is not None:
                filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        query = (
            self.get_query()
                .PROCESSED_FRINGE()
                .add_filters(*filters)
                .match_descriptors(Header.instrument,
                                   Ghost.detector_x_bin,
                                   Ghost.detector_y_bin,
                                   Ghost.filter_name)
                # Absolute time separation must be within 1 year
                .max_interval(days=365)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    # We don't handle processed ones (yet)
    @not_processed
    @not_imaging
    def spectwilight(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the best spectwilight - ie spectroscopy twilight
        ie MOS / IFU / LS twilight

        This will find GHOST spec twilights matching the amp read area, filter name, disperser,
        filter plane mask, and x and y binning.  It matches central wavelength within 0.02 microns.
        It matches within 1 year.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw spec twilights
        howmany : int, default 2
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # Default number to associate
        howmany = howmany if howmany else 2

        filters = []
        # The science amp_read_area must be equal or substring of the cal amp_read_area
        # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # have a subset of the amps thus we must do the substring match
        if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
            filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        elif self.descriptors['amp_read_area'] is not None:
                filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        query = (
            self.get_query()
                # They are OBJECT spectroscopy frames with target twilight
                .raw().OBJECT().spectroscopy(True).object('Twilight')
                .add_filters(*filters)
                .match_descriptors(Header.instrument,
                                   Ghost.detector_x_bin,
                                   Ghost.detector_y_bin,
                                   Ghost.filter_name,
                                   Ghost.disperser,
                                   Ghost.focal_plane_mask)
                # Must match central wavelength to within some tolerance.
                # We don't do separate ones for dithers in wavelength?
                # tolerance = 0.02 microns
                .tolerance(central_wavelength=0.02)
                # Absolute time separation must be within 1 year
                .max_interval(days=365)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)


    # We don't handle processed ones (yet)
    @not_processed
    @not_imaging
    def specphot(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the best specphot observation

        This will find GHOST spec photometry matching the amp read area, filter name, and disperser.
        The data must be partnerCal or progCal and not be Twilight.  If the focal plane mask is measured
        in arcsec, it will match the central wavelength to within 0.1 microns, else it matches within 0.05
        microns.

        It matches within 1 year.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw spec photometry
        howmany : int, default 2
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # Default number to associate
        howmany = howmany if howmany else 4

        filters = []
        # Must match the focal plane mask, unless the science is a mos mask in which case the specphot is longslit
        if 'MOS' in self.types:
            filters.append(Ghost.focal_plane_mask.contains('arcsec'))
            tol = 0.10 # microns
        else:
            filters.append(Ghost.focal_plane_mask == self.descriptors['focal_plane_mask'])
            tol = 0.05 # microns

        # The science amp_read_area must be equal or substring of the cal amp_read_area
        # If the science frame uses all the amps, then they must be a direct match as all amps must be there
        # - this is more efficient for the DB as it will use the index. Otherwise, the science frame could
        # have a subset of the amps thus we must do the substring match
        if self.descriptors['detector_roi_setting'] in ['Full Frame', 'Central Spectrum']:
            filters.append(Ghost.amp_read_area == self.descriptors['amp_read_area'])
        elif self.descriptors['amp_read_area'] is not None:
                filters.append(Ghost.amp_read_area.contains(self.descriptors['amp_read_area']))

        query = (
            self.get_query()
                # They are OBJECT partnerCal or progCal spectroscopy frames with target not twilight
                .raw().OBJECT().spectroscopy(True)
                .add_filters(Header.observation_class.in_(['partnerCal', 'progCal']),
                             Header.object != 'Twilight',
                             *filters)
                # Found lots of examples where detector binning does not match, so we're not adding those
                .match_descriptors(Header.instrument,
                                   Ghost.filter_name,
                                   Ghost.disperser)
                # Must match central wavelength to within some tolerance.
                # We don't do separate ones for dithers in wavelength?
                .tolerance(central_wavelength=tol)
                # Absolute time separation must be within 1 year
                .max_interval(days=365)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    # We don't handle processed ones (yet)
    @not_processed
    @not_spectroscopy
    def photometric_standard(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the best phot_std observation

        This will find GHOST photometric standards matching the filter name.  It must be a partnerCal with
        a CAL program id.  This matches within 1 day.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw photometric standards
        howmany : int, default 4
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # Default number to associate
        howmany = howmany if howmany else 4

        query = (
            self.get_query()
                # They are OBJECT imaging partnerCal frames taken from CAL program IDs
                .photometric_standard(OBJECT=True, partnerCal=True)
                .add_filters(Header.program_id.like('G_-CAL%'))
                .match_descriptors(Header.instrument,
                                   Ghost.filter_name)
                # Absolute time separation must be within 1 days
                .max_interval(days=1)
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)

    # We don't handle processed ones (yet)
    @not_processed
    def mask(self, processed=False, howmany=None, return_query=False):
        """
        Method to find the MASK (MDF) file

        This will find GHOST masks matching the focal plane mask.

        Parameters
        ----------

        processed : bool
            Indicate if we want to retrieve processed or raw masks
        howmany : int, default 1
            How many do we want results

        Returns
        -------
            list of :class:`fits_storage.orm.header.Header` records that match the criteria
        """
        # Default number to associate
        howmany = howmany if howmany else 1

        query = (
            self.get_query()
                # They are MASK observation type
                # The focal_plane_mask of the science file must match the data_label of the MASK file (yes, really...)
                # Cant force an instrument match as sometimes it just says GHOST in the mask...
                .add_filters(Header.observation_type == 'MASK',
                             Header.data_label == self.descriptors['focal_plane_mask'],
                             Header.instrument.startswith('GHOST'))
            )
        if return_query:
            return query.all(howmany), query
        else:
            return query.all(howmany)
