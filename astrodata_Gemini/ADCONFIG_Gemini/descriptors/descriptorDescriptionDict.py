# Use the descriptorDescDict dictionary to fill in the doc string for a
# descriptor that requires more details than the standard doc string as
# defined in mkCalculatorInterface:
#
#     Return the <descriptor> value
#     :param dataset: the data set
#     :type dataset: AstroData
#     :rtype: string
#     :return: the <descriptor> for the observation
#
# Add to this dictionary using the format <descriptor>:<doc string>,

descriptorDescDict = {
    # central_wavelength
    'central_wavelength':'Return the central_wavelength value\n' +
    '        :param dataset: the data set\n' +
    '        :type dataset: AstroData\n' +
    '        :param asMicrometers: set to True to return the ' +
    'central_wavelength \n                              value in units of ' +
    'Micrometers\n' +
    '        :type asMicrometers: Python boolean\n' +
    '        :param asNanometers: set to True to return the ' +
    'central_wavelength \n                             value in units of ' +
    'Nanometers\n' +
    '        :type asNanometers: Python boolean\n' +
    '        :param asAngstroms: set to True to return the ' +
    'central_wavelength \n                            value in units of ' +
    'Angstroms\n' +
    '        :type asAngstroms: Python boolean\n' +
    '        :param asDict: set to True to return a dictionary, where the ' +
    'number of \n                       dictionary elements equals the ' +
    'number of pixel data \n                       extensions in the image. ' +
    'The key of the dictionary is \n                       an (EXTNAME, ' +
    'EXTVER) tuple, if available. Otherwise, \n                       the ' +
    'key is the integer index of the extension.\n' + \
    '        :type asDict: Python boolean\n' + \
    '        :rtype: dictionary containing one or more float(s)\n' + \
    '        :return: the central wavelength (in meters by default) of the ' +
    '\n                 observation',
    # data_section
    'data_section':'Return the data_section value\n' +
    '        :param dataset: the data set\n' +
    '        :type dataset: AstroData\n' +
    '        :param pretty: set to True to return a human meaningful ' +
    'data_section \n                       value in the form [x1:x2,y1:y2] ' +
    'that uses 1-based \n                       indexing\n' +
    '        :type pretty: Python boolean\n' +
    '        :param asDict: set to True to return a dictionary, where the ' +
    'number of \n                       dictionary elements equals the ' +
    'number of pixel data \n                       extensions in the image. ' +
    'The key of the dictionary is \n                       an (EXTNAME, ' +
    'EXTVER) tuple, if available. Otherwise, \n                       the ' +
    'key is the integer index of the extension.\n' + \
    '        :type asDict: Python boolean\n' + \
    '        :rtype: dictionary containing one or more tuple(s) that use ' +
    '0-based \n                indexing in the form ' + \
    '(x1 - 1, x2 - 1, y1 - 1, y2 - 1) as \n                default, or one ' +
    'or more string(s) that use 1-based indexing in \n                the ' +
    'form [x1:x2,y1:y2] if pretty=True, where x1, x2, y1 and y2 ' +
    '\n                are integers.\n' + \
    '        :return: the section of the data of the observation',
    # detector_section
    'detector_section':'Return the detector_section value\n' +
    '        :param dataset: the data set\n' +
    '        :type dataset: AstroData\n' +
    '        :param pretty: set to True to return a human meaningful \n' +
    '                       detector_section value in the form ' +
    '[x1:x2,y1:y2] that \n                       uses 1-based indexing\n' +
    '        :type pretty: Python boolean\n' +
    '        :param asDict: set to True to return a dictionary, where the ' +
    'number of \n                       dictionary elements equals the ' +
    'number of pixel data \n                       extensions in the image. ' +
    'The key of the dictionary is \n                       an (EXTNAME, ' +
    'EXTVER) tuple, if available. Otherwise, \n                       the ' +
    'key is the integer index of the extension.\n' + \
    '        :type asDict: Python boolean\n' + \
    '        :rtype: dictionary containing one or more tuple(s) that use ' +
    '0-based \n                indexing in the form ' + \
    '(x1 - 1, x2 - 1, y1 - 1, y2 - 1) as \n                default, or one ' +
    'or more string(s) that use 1-based indexing in \n                the ' +
    'form [x1:x2,y1:y2] if pretty=True, where x1, x2, y1 and y2 ' +
    '\n                are integers.\n' + \
    '        :return: the detector section of the observation',
    # dispersion
    'dispersion':'Return the dispersion value\n' +
    '        :param dataset: the data set\n' +
    '        :type dataset: AstroData\n' +
    '        :param asMicrometers: set to True to return the ' +
    'dispersion \n                              value in units of ' +
    'Micrometers\n' +
    '        :type asMicrometers: Python boolean\n' +
    '        :param asNanometers: set to True to return the ' +
    'dispersion \n                             value in units of ' +
    'Nanometers\n' +
    '        :type asNanometers: Python boolean\n' +
    '        :param asAngstroms: set to True to return the ' +
    'dispersion \n                            value in units of ' +
    'Angstroms\n' +
    '        :type asAngstroms: Python boolean\n' +
    '        :param asDict: set to True to return a dictionary, where the ' +
    'number of \n                       dictionary elements equals the ' +
    'number of pixel data \n                       extensions in the image. ' +
    'The key of the dictionary is \n                       an (EXTNAME, ' +
    'EXTVER) tuple, if available. Otherwise, \n                       the ' +
    'key is the integer index of the extension.\n' + \
    '        :type asDict: Python boolean\n' + \
    '        :rtype: dictionary containing one or more float(s)\n' + \
    '        :return: the dispersion (in meters per pixel by default) of ' +
    'the \n                 observation',
    # ut_datetime
    'ut_datetime':'Return the ut_datetime value\n' +
    '        This descriptor attempts to figure out the datetime even when ' +
    'the\n        headers are malformed or not present. It tries just about ' +
    'every header\n        combination that could allow it to determine an ' +
    'appropriate datetime\n        for the file in question. This makes it ' +
    'somewhat specific to Gemini\n        data, in that the headers it ' +
    'looks at, and the assumptions it makes in\n        trying to parse ' +
    'their values, are those known to occur in Gemini data.\n        Note ' +
    'that some of the early gemini data, and that taken from lower\n        ' +
    'level engineering interfaces, lack standard headers. Also the format\n' +
    '        and occurence of various headers has changed over time, even ' +
    'on the\n        same instrument. If strict is set to True, the date or ' +
    'time are\n        determined from valid FITS keywords. If it cannot be ' +
    'determined, None\n        is returned. If dateonly or timeonly are set ' +
    'to True, then a\n        datetime.date object or datetime.time object, ' +
    'respectively, is\n        returned, containing only the date or time, ' +
    'respectively. These two\n        interplay with strict in the sense ' +
    'that if strict is set to True and a\n        date can be determined ' +
    'but not a time, then this function will return\n        None unless ' +
    'the dateonly flag is set, in which case it will return the\n        ' +
    'valid date. The dateonly and timeonly flags are intended for use by\n' +
    '        the ut_date and ut_time descriptors.\n' +
    '        :param dataset: the data set\n' +
    '        :type dataset: AstroData\n' +
    '        :param strict: set to True to not try to guess the date or ' +
    'time\n' +
    '        :type strict: Python boolean\n' +
    '        :param dateonly: set to True to return a datetime.date\n' +
    '        :type dateonly: Python boolean\n' +
    '        :param timeonly: set to True to return a datetime.time\n' +
    '        :param timeonly: Python boolean\n' +
    '        :rtype: datetime.datetime (dateonly=False and timeonly=False)\n' +
    '        :rtype: datetime.time (timeonly=True)\n' +
    '        :rtype: datetime.date (dateonly=True)\n' +
    '        :return: the UT date and time at the start of the observation',
                      }

# Use the returnTypeDict dictionary to change the return type of a descriptor
# when using the standard doc string. The default return type is 'string'. Add
# to this dictionary using the format <descriptor>:<return type>,

returnTypeDict = {
    'airmass':'float',
    'azimuth':'float',
    'cass_rotator_pa':'float',
    'central_wavelength':'float',
    'coadds':'integer',
    'dec':'float',
    'detector_x_bin':'integer',
    'detector_y_bin':'integer',
    'dispersion_axis':'integer',
    'elevation':'float',
    'exposure_time':'float',
    'gain':'float',
    'mdf_row_id':'integer',
    'nod_count':'integer',
    'nod_pixels':'integer',
    'non_linear_level':'integer',
    'pixel_scale':'float',
    'ra':'float',
    'read_noise':'float',
    'saturation_level':'integer',
    'ut_date':'datatime.date',
    'ut_time':'datatime.time',
    'wavelength_reference_pixel':'float',
    'x_offset':'float',
    'y_offset':'float',
                 }

# Use the detailedNameDict dictionary to use a more detailed description of
# the descriptor in the return field (include units if applicable) when using
# the standard doc string. The default is the descriptor name. Add to this
# dictionary using the format <descriptor>:<detailed descriptor description>,

detailedNameDict = {
    'airmass':'mean airmass of the observation',
    'amp_read_area':'composite string containing the name of the detector\n' +
        '                 amplifier (ampname) and the readout area of the ' +
        'CCD (detsec) \n                 used for the observation',
    'azimuth':'azimuth (in degrees between 0 and 360) of the observation',
    'camera':'camera used for the observation',
    'cass_rotator_pa':'cassegrain rotator position angle (in degrees between '+
        '-360\n                 and 360) of the observation',
    'coadds':'number of coadds used for the observation',
    'data_label':'data label of the observation',
    'dec':'declination (in decimal degrees) of the observation',
    'decker':'decker position used for the observation',
    'detector_x_bin':'binning of the x-axis of the detector used for the ' +
        '\n                 observation',
    'detector_y_bin':'binning of the y-axis of the detector used for the ' +
        '\n                 observation',
    'disperser':'disperser used for the observation',
    'dispersion_axis':'dispersion axis (x = 1; y = 2; z = 3) of the ' +
        'observation',
    'elevation':'elevation (in degrees) of the observation',
    'exposure_time':'total exposure time (in seconds) of the observation',
    'filter_name':'unique, sorted filter name idenifier string used for ' +
    'the \n                 observation',
    'focal_plane_mask':'focal plane mask used for the observation',
    'gain':'gain (in electrons per ADU) of the observation',
    'grating':'grating used for the observation',
    'gain_setting':'gain setting of the observation',
    'instrument':'instrument used for the observation',
    'local_time':'local time (in HH:MM:SS.S) at the start of the observation',
    'mdf_row_id':'corresponding reference row in the MDF',
    'nod_count':'number of nod and shuffle cycles in the nod and shuffle \n' +
        '                 observation',
    'nod_pixels':'number of pixel rows the charge is shuffled by in the ' +
        'nod \n                 and shuffle observation',
    'non_linear_level':'non linear level in the raw images (in ADU) of the ' +
        '\n                 observation',
    'object':'name of the target object observed',
    'observation_class':'class (either \'science\', \'progCal\', ' +
        '\'partnerCal\', \'acq\', \n                 \'acqCal\' or ' +
        '\'dayCal\') of the observation',
    'observation_epoch':'epoch (in years) at the start of the observation',
    'observation_id':'ID (e.g., GN-2011A-Q-123-45) of the observation',
    'observation_type':'type (either \'OBJECT\', \'DARK\', \'FLAT\', ' +
    '\'ARC\', \'BIAS\' or \n                 \'MASK\') of the observation',
    'pixel_scale':'pixel scale (in arcsec per pixel) of the observation',
    'prism':'prism used for the observation',
    'program_id':'Gemini program ID (e.g., GN-2011A-Q-123) of the \n' +
    '                 observation',
    'pupil_mask':'pupil mask used for the observation',
    'qa_state':'quality assessment state (either \'Undefined\', \'Pass\', \n' +
        '                 \'Usable\', \'Fail\' or \'CHECK\') of the ' +
        'observation',
    'ra':'Right Ascension (in decimal degrees) of the observation',
    'raw_bg':'raw background (either \'20-percentile\', \'50-percentile\', ' +
        '\n                 \'80-percentile\' or \'Any\') of the observation',
    'raw_cc':'raw cloud cover (either \'50-percentile\', \'70-percentile\', ' +
        '\n                 \'80-percentile\', \'90-percentile\' or ' +
        '\'Any\') of the observation',
    'raw_iq':'raw image quality (either \'20-percentile\', \n' +
        '                 \'70-percentile\', \'85-percentile\' or ' +
        '\'Any\') of the observation',
    'raw_wv':'raw water vapour (either \'20-percentile\', \n' +
        '                 \'50-percentile\', \'80-percentile\' or \'Any\') ' +
        'of the observation',
    'read_mode':'read mode (either \'Very Faint Objects\', \n' +
        '                 \'Faint Object(s)\', \'Medium Object\', ' +
        '\'Bright Object(s)\', \n                 \'Very Bright Object\', ' +
        '\'Low Background\', \'Medium Background\', \n                 ' +
        '\'High Background\' or \'Invalid\') of the observation',
    'read_noise':'estimated readout noise (in electrons) of the observation',
    'read_speed_setting':'read speed setting (either \'fast\' or \'slow\') ' +
        'of the \n                 observation',
    'saturation_level':'saturation level in the raw images (in ADU) of the ' +
        '\n                 observation',
    'slit':'slit used for the observation',
    'telescope':'telescope used for the observation',
    'ut_date':'UT date at the start of the observation',
    'ut_time':'UT time at the start of the observation',
    'wavefront_sensor':'wavefront sensor (either \'AOWFS\', \'OIWFS\', ' +
        '\'PWFS1\', \n                 \'PWFS2\', some combination in ' +
        'alphebetic order separated with \n                 an ampersand or ' +
        'None) used for the observation',
    'wavelength_reference_pixel':'reference pixel of the central wavelength ' +
        'of the \n                 observation',
    'well_depth_setting':'well depth setting (either \'Shallow\', \'Deep\' ' +
        'or \n                 \'Invalid\') of the observation',
    'x_offset':'x offset of the observation',
    'y_offset':'y offset of the observation',
                   }

# Use the asDictArgDict dictionary to change the standard doc string of a
# descriptor to include the asDict argument description for those descriptors
# that access keywords in the headers of the pixel data extensions. The
# default return type is 'no'. Add to this dictionary using the format
# <descriptor>:'yes',

asDictArgDict = {
    'amp_read_area':'yes',
    'detector_x_bin':'yes',
    'detector_y_bin':'yes',
    'gain':'yes',
    'mdf_row_id':'yes',
    'read_noise':'yes',
    'wavelength_reference_pixel':'yes',
                }

# Use the stripIDArgDict dictionary to change the standard doc string of a
# descriptor to include the stripID and pretty argument descriptions for those
# descriptors that return values with a Gemini ID associated with them. The
# default return type is 'no'. Add to this dictionary using the format
# <descriptor>:'yes',

stripIDArgDict = {
    'decker':'yes',
    'disperser':'yes',
    'filter_name':'yes',
    'focal_plane_mask':'yes',
    'grating':'yes',
    'prism':'yes',
    'slit':'yes',
                 }
