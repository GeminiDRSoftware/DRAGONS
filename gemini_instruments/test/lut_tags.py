fixture_data = {
    # GMOS-Level Tags
    #   Tags: GMOS, BIAS, DARK, FRINGE, FLAT (whether it's GCAL or TWILIGT), GCALFLAT, TWILIGHT, ARC, IMAGE, SPECT, IFU, LS, MOS, NODANDSHUFFLE, BLUE, RED, TWO
    #
    #   GMOS Bias
    ('GMOS', 'N20110524S0358.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'GMOS', 'BIAS', 'CAL', 'NON_SIDEREAL',
                                      'RAW', 'UNPREPARED', 'AT_ZENITH'],
    ('GMOS', 'N20110524S0358_bias.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'GMOS', 'BIAS', 'CAL',
                                           'NON_SIDEREAL', 'OVERSCAN_SUBTRACTED', 'OVERSCAN_TRIMMED', 'PREPARED',
                                           'PROCESSED', 'AT_ZENITH'],
    ('GMOS', 'S20110627S0151.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GEMINI', 'SOUTH', 'GMOS', 'BIAS', 'CAL',
                                      'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('GMOS', 'S20070318S0274_bias.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GEMINI', 'SOUTH', 'GMOS', 'BIAS',
                                           'NON_SIDEREAL', 'OVERSCAN_SUBTRACTED', 'OVERSCAN_TRIMMED', 'PREPARED',
                                           'PROCESSED'],

    #   GMOS Dark
    ('GMOS', 'N20160106S0653.fits'): ['AZEL_TARGET', 'GEMINI', 'NORTH', 'GMOS', 'DARK', 'RAW', 'NON_SIDEREAL',
                                      'UNPREPARED', 'AT_ZENITH', 'CAL', 'MOS'],
    ('GMOS', 'S20160725S0008.fits'): ['GEMINI', 'SOUTH', 'GMOS', 'DARK', 'RAW', 'SIDEREAL', 'UNPREPARED',
                                      'NODANDSHUFFLE', 'CAL'],

    #   GMOS Arc
    ('GMOS', 'S20160724S0218.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GMOS', 'LS', 'ARC', 'RAW', 'SPECT', 'SIDEREAL',
                                      'UNPREPARED'],
    ('GMOS', 'N20160721S0351.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'GMOS', 'MOS', 'ARC', 'RAW',
                                      'SPECT', 'NON_SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'N20160706S0220.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'ARC', 'ONESLIT_RED', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'S20060131S0131.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GMOS', 'IFU', 'ARC', 'ONESLIT_BLUE', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'N20160421S0217.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'ARC', 'TWOSLIT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'N20020829S0026.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'MOS', 'ARC', 'NODANDSHUFFLE', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],

    #   GMOS Fringe
    # Those will only be recognized as PROCESSED FRINGE.  As raw, they
    # cannot be recognized.
    ('GMOS', 'N20110927S0170_fringe.fits'): ['FRINGE', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'OVERSCAN_SUBTRACTED',
                                             'OVERSCAN_TRIMMED', 'PREPARED', 'PROCESSED', 'SIDEREAL', 'CAL'],

    #   GMOS Imaging Data
    ('GMOS', 'N20120203S0284.fits'): ['GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'SIDEREAL', 'RAW', 'UNPREPARED'],
    ('GMOS', 'S20111120S0478.fits'): ['GEMINI', 'SOUTH', 'GMOS', 'IMAGE', 'SIDEREAL', 'RAW', 'UNPREPARED'],
    ('GMOS', 'N20120121S0175.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'SIDEREAL', 'RAW',
                                      'UNPREPARED'],
    ('GMOS', 'N20110718S0294.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'FLAT', 'TWILIGHT', 'SIDEREAL', 'RAW',
                                      'UNPREPARED'],
    ('GMOS', 'N20110718S0290_flat.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'FLAT', 'TWILIGHT', 'SIDEREAL',
                                           'OVERSCAN_SUBTRACTED', 'OVERSCAN_TRIMMED', 'PREPARED', 'PROCESSED'],
    ('GMOS', 'S20150903S0061.fits'): ['ACQUISITION', 'GEMINI', 'SOUTH', 'GMOS', 'IMAGE', 'SIDEREAL', 'RAW',
                                      'UNPREPARED'],
    ('GMOS', 'S20140625S0361.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GMOS', 'IMAGE', 'FLAT', 'TWILIGHT', 'SIDEREAL', 'RAW',
                                      'UNPREPARED'],

    #   GMOS Longslit Data
    ('GMOS', 'N20160726S0179.fits'): ['GEMINI', 'NORTH', 'GMOS', 'LS', 'RAW', 'SPECT', 'SIDEREAL', 'UNPREPARED',
                                      'STANDARD', 'CAL'],
    ('GMOS', 'S20091217S0032.fits'): ['GEMINI', 'SOUTH', 'GMOS', 'LS', 'NODANDSHUFFLE', 'RAW', 'SPECT', 'SIDEREAL',
                                      'UNPREPARED'],
    ('GMOS', 'N20160726S0180.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'LS', 'FLAT', 'GCALFLAT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'N20160722S0284.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'LS', 'SLITILLUM', 'TWILIGHT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    # This one is not working in the current system.  Doesn't detect
    # it as TWILIGHT, nor FLAT.
    ('GMOS', 'S20160722S0210.fits'): ['ACQUISITION', 'GEMINI', 'SOUTH', 'GMOS', 'IMAGE', 'RAW', 'SIDEREAL',
                                      'UNPREPARED', 'LS', 'THRUSLIT'],
    # New tag: MASK.  I think that
    # the conditions will be GRATING='MIRROR' and MASKTYP=1.

    #   GMOS MOS Data
    ('GMOS', 'N20110826S0336.fits'): ['GEMINI', 'NORTH', 'GMOS', 'MOS', 'RAW', 'SPECT', 'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'S20150417S0092.fits'): ['GEMINI', 'SOUTH', 'GMOS', 'MOS', 'NODANDSHUFFLE', 'RAW', 'SPECT', 'SIDEREAL',
                                      'UNPREPARED'],
    ('GMOS', 'N20160708S0096.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'MOS', 'FLAT', 'GCALFLAT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    ('GMOS', 'N20160605S0099.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'MOS', 'SLITILLUM', 'TWILIGHT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED'],
    # This one is not working in the current system.  Doesn't detect
    # it as TWILIGHT, nor FLAT.
    ('GMOS', 'N20160708S0085.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'RAW', 'SIDEREAL',
                                      'UNPREPARED', 'THRUSLIT', 'MOS'],
    # New tag: MASK.  I think that
    # the conditions will be GRATING == 'MIRROR' and MASKTYP != 0.

    #   GMOS IFU Data
    ('GMOS', 'N20160706S0218.fits'): ['GEMINI', 'NORTH', 'GMOS', 'IFU', 'ONESLIT_RED', 'RAW', 'SPECT', 'SIDEREAL',
                                      'UNPREPARED'],
    ('GMOS', 'S20060130S0079.fits'): ['GEMINI', 'SOUTH', 'GMOS', 'IFU', 'NODANDSHUFFLE', 'RAW', 'SPECT', 'SIDEREAL',
                                      'UNPREPARED', 'TWOSLIT'],
    # The two-slit nature is not identified in the old system.
    # MASKNAME='IFU-NS-2' in that file.  the rule has 'IFU-2-NS', that
    # appears to be wrong.
    ('GMOS', 'N20160716S0148.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'FLAT', 'ONESLIT_RED', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED', 'GCALFLAT'],
    ('GMOS', 'S20060131S0117.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GMOS', 'IFU', 'FLAT', 'ONESLIT_BLUE', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED', 'GCALFLAT'],
    ('GMOS', 'N20160524S0119.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'FLAT', 'TWOSLIT', 'RAW', 'SPECT',
                                      'SIDEREAL', 'UNPREPARED', 'GCALFLAT'],
    ('GMOS', 'N20160722S0285.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'SLITILLUM', 'TWILIGHT', 'RAW', 'TWOSLIT',
                                      'SPECT', 'SIDEREAL', 'UNPREPARED'],
    # This one is not working in the current system.  Doesn't detect
    # it as TWILIGHT, nor FLAT.
    ('GMOS', 'N20160620S0373.fits'): ['CAL', 'GEMINI', 'NORTH', 'GMOS', 'IFU', 'SLITILLUM', 'TWILIGHT', 'RAW', 'ONESLIT_RED',
                                      'SPECT', 'SIDEREAL', 'UNPREPARED'],
    # This one is not working in the current system.  Doesn't detect
    # it as TWILIGHT, nor FLAT.
    ('GMOS', 'S20051226S0033.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GMOS', 'IFU', 'SLITILLUM', 'TWILIGHT', 'RAW',
                                      'ONESLIT_BLUE', 'SPECT', 'SIDEREAL', 'UNPREPARED'],
    # This one is not working in the current system.  Doesn't detect
    # it as TWILIGHT, nor FLAT.
    ('GMOS', 'N20160620S0317.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GMOS', 'IMAGE', 'RAW', 'SIDEREAL',
                                      'UNPREPARED', 'ONESLIT_RED', 'IFU', 'THRUSLIT'],
    # For tag MASK: GRATING == 'MIRROR' AND MASKTYP != 0

    #   GMOS N&S Data
    # See in longslit, MOS, and IFU.

    # NIRI Data
    #   NIRI Darks
    ('NIRI', 'N20131215S0503.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'NIRI', 'DARK', 'NON_SIDEREAL', 'RAW',
                                      'UNPREPARED', 'AT_ZENITH'],
    ('NIRI', 'N20130409S0105_dark.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'NIRI', 'DARK',
                                           'NON_SIDEREAL', 'PREPARED', 'PROCESSED'],

    #   NIRI Imaging
    # FIXME: Not found in archive.
    # ('NIRI', 'N20131215S0178_prepared.fits'): ['GEMINI', 'NORTH', 'IMAGE', 'NIRI', 'PREPARED', 'SIDEREAL'],
    ('NIRI', 'N20131215S0178.fits'): ['GEMINI', 'NORTH', 'IMAGE', 'NIRI', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('NIRI', 'N20131215S0427.fits'): ['AZEL_TARGET', 'CAL', 'GCAL_IR_OFF', 'GEMINI', 'NORTH', 'IMAGE', 'NIRI', 'FLAT',
                                      'GCALFLAT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'LAMPOFF', 'AT_ZENITH'],
    ('NIRI', 'N20131215S0428.fits'): ['AZEL_TARGET', 'CAL', 'GCAL_IR_ON', 'GEMINI', 'NORTH', 'IMAGE', 'NIRI', 'FLAT',
                                      'GCALFLAT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'LAMPON', 'AT_ZENITH'],
    ('NIRI', 'N20130404S0470.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GCAL_IR_OFF', 'GEMINI', 'NORTH', 'IMAGE',
                                      'NIRI', 'FLAT', 'GCALFLAT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'LAMPOFF'],

    #   NIRI Longslit
    ('NIRI', 'N20120505S0147.fits'): ['GEMINI', 'NORTH', 'NIRI', 'LS', 'SPECT', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('NIRI', 'N20100620S0126.fits'): ['GEMINI', 'NORTH', 'NIRI', 'LS', 'ARC', 'SPECT', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    # Arc is not defined for NIRI in the old system.  (How can that be?)
    ('NIRI', 'N20120505S0564.fits'): ['AZEL_TARGET', 'CAL', 'GCAL_IR_OFF', 'LAMPOFF', 'FLAT', 'GCALFLAT', 'GEMINI',
                                      'NORTH', 'LS', 'NIRI', 'SPECT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'AT_ZENITH'],
    ('NIRI', 'N20120505S0552.fits'): ['AZEL_TARGET', 'CAL', 'GCAL_IR_ON', 'LAMPON', 'FLAT', 'GCALFLAT', 'GEMINI',
                                      'NORTH', 'LS', 'NIRI', 'SPECT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'AT_ZENITH'],

    # GNIRS Data
    #   GNIRS Darks
    ('GNIRS', 'N20140812S0150.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'GNIRS', 'DARK', 'NON_SIDEREAL', 'RAW',
                                       'UNPREPARED', 'AT_ZENITH'],
    ('GNIRS', 'N20140812S0142_dark.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'NORTH', 'GNIRS', 'DARK', 'NON_SIDEREAL',
                                            'PREPARED', 'PROCESSED', 'AT_ZENITH'],

    #   GNIRS Imaging
    ('GNIRS', 'N20120117S0019.fits'): ['GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('GNIRS', 'N20140717S0232.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'RAW', 'SIDEREAL',
                                       'UNPREPARED', 'THRUSLIT'],
    ('GNIRS', 'N20120117S0041.fits'): ['CAL', 'GCAL_IR_ON', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'FLAT', 'GCALFLAT',
                                       'RAW', 'SIDEREAL', 'UNPREPARED', 'LAMPON'],
    ('GNIRS', 'N20120117S0042.fits'): ['CAL', 'GCAL_IR_OFF', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'FLAT', 'GCALFLAT',
                                       'RAW', 'SIDEREAL', 'UNPREPARED', 'LAMPOFF'],
    ('GNIRS', 'N20131222S0070_flat.fits'): ['CAL', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'FLAT', 'GCALFLAT', 'PREPARED',
                                            'PROCESSED', 'SIDEREAL'],

    #   GNIRS Longslit
    ('GNIRS', 'N20160720S0096.fits'): ['CAL', 'GCAL_IR_ON', 'LAMPON', 'FLAT', 'GCALFLAT', 'GEMINI', 'NORTH', 'LS',
                                       'GNIRS', 'SPECT', 'SIDEREAL', 'RAW', 'UNPREPARED'],
    # Apparently GNIRS has no lamp off for spectroscopy.
    ('GNIRS', 'N20130814S0231.fits'): ['CAL', 'GEMINI', 'NORTH', 'GNIRS', 'LS', 'ARC', 'SPECT', 'RAW', 'SIDEREAL',
                                       'UNPREPARED'],
    ('GNIRS', 'N20160624S0263.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'RAW', 'SIDEREAL',
                                       'UNPREPARED', 'THRUSLIT'],
    # for MASK: SLIT does not contain "Acq"
    ('GNIRS', 'N20160624S0264.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'GNIRS', 'IMAGE', 'RAW', 'SIDEREAL',
                                       'UNPREPARED'],

    #    GNIRS Cross-dispersed
    #    ('GNIRS', 'N20110826S0155.fits') : ['GEMINI', 'NORTH', 'GNIRS', 'XD', 'SPECT', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('GNIRS', 'N20160726S0252.fits'): ['GEMINI', 'NORTH', 'GNIRS', 'XD', 'SPECT', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    # XD tag: DECKER contains 'XD'
    ('GNIRS', 'N20160726S0260.fits'): ['CAL', 'GEMINI', 'NORTH', 'GNIRS', 'XD', 'SPECT', 'RAW', 'SIDEREAL',
                                       'UNPREPARED', 'FLAT', 'GCALFLAT', 'LAMPON', 'GCAL_IR_ON'],
    ('GNIRS', 'N20160726S0261.fits'): ['CAL', 'GEMINI', 'NORTH', 'GNIRS', 'XD', 'SPECT', 'RAW', 'SIDEREAL',
                                       'UNPREPARED', 'FLAT', 'GCALFLAT', 'LAMPOFF', 'GCAL_IR_OFF'],
    # this one is with the QH lamp.  not sure we need a tag for it.
    ('GNIRS', 'N20160722S0232.fits'): ['CAL', 'GEMINI', 'NORTH', 'GNIRS', 'XD', 'SPECT', 'RAW', 'SIDEREAL',
                                       'UNPREPARED', 'ARC'],
    # tag ARC:  OBSTYPE == 'ARC'
    ('GNIRS', 'N20160726S0301.fits'): ['LAMPOFF', 'CAL', 'GEMINI', 'AT_ZENITH', 'SPECT', 'UNPREPARED', 'XD', 'GNIRS',
                                       'NON_SIDEREAL', 'PINHOLE', 'RAW', 'FLAT', 'AZEL_TARGET', 'NORTH', 'GCAL_IR_OFF'],
    # tag PINHOLE: SLIT contains 'Pinholes'

    #    GNIRS IFU

    # NIFS Data
    #   NIFS Darks
    ('NIFS', 'N20160727S0077.fits'): ['CAL', 'AZEL_TARGET', 'GEMINI', 'NORTH', 'DARK', 'NIFS', 'NON_SIDEREAL', 'RAW',
                                      'UNPREPARED', 'AT_ZENITH'],

    #   NIFS IFU
    ('NIFS', 'N20110707S0196.fits'): ['GEMINI', 'NORTH', 'IFU', 'NIFS', 'SPECT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('NIFS', 'N20160708S0134.fits'): ['GEMINI', 'NORTH', 'IFU', 'NIFS', 'SPECT', 'SIDEREAL', 'RAW', 'UNPREPARED'],
    ('NIFS', 'N20160727S0078.fits'): ['CAL', 'AZEL_TARGET', 'GEMINI', 'NORTH', 'ARC', 'NIFS', 'IFU', 'SPECT',
                                      'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'AT_ZENITH'],
    ('NIFS', 'N20160727S0079.fits'): ['CAL', 'AZEL_TARGET', 'GEMINI', 'NORTH', 'FLAT', 'GCALFLAT', 'LAMPON',
                                      'GCAL_IR_ON', 'IFU', 'NIFS', 'SPECT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('NIFS', 'N20160708S0119.fits'): ['ACQUISITION', 'GEMINI', 'NORTH', 'IFU', 'NIFS', 'SPECT', 'RAW', 'SIDEREAL',
                                      'SPECT', 'UNPREPARED'],
    ('NIFS', 'N20160705S0221.fits'): ['CAL', 'RONCHI', 'AZEL_TARGET', 'GEMINI', 'NORTH', 'IFU', 'NIFS', 'SPECT',
                                      'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'AT_ZENITH', 'GCAL_IR_OFF', 'LAMPOFF',
                                      'GCALFLAT', 'FLAT'],

    # F2 Data
    #   F2 Darks
    ('F2', 'S20150516S0373.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'F2', 'DARK', 'GEMINI', 'SOUTH', 'NON_SIDEREAL',
                                    'RAW', 'UNPREPARED'],
    ('F2', 'S20141031S0274_dark.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'F2', 'DARK', 'GEMINI', 'SOUTH',
                                         'NON_SIDEREAL', 'PREPARED', 'PROCESSED'],

    #   F2 Imaging
    ('F2', 'S20150511S0168.fits'): ['F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20100122S0063.fits'): ['F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20150511S0274.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'F2', 'IMAGE', 'FLAT', 'GCALFLAT', 'GCAL_IR_ON',
                                    'GEMINI', 'SOUTH', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'LAMPON'],
    ('F2', 'S20150511S0275.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'F2', 'IMAGE', 'FLAT', 'GCALFLAT', 'GCAL_IR_OFF',
                                    'GEMINI', 'SOUTH', 'NON_SIDEREAL', 'RAW', 'UNPREPARED', 'LAMPOFF'],
    # FIXME: Crashes when downloading.
    # ('F2', 'S20141107S0062_flat.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'F2', 'IMAGE', 'FLAT', 'GCALFLAT', 'GEMINI',
    #                                      'SOUTH', 'NON_SIDEREAL', 'PREPARED', 'PROCESSED'],

    #   F2 Longslit
    ('F2', 'S20160724S0128.fits'): ['F2', 'LS', 'SPECT', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20160724S0130.fits'): ['CAL', 'F2', 'LS', 'FLAT', 'GCALFLAT', 'SPECT', 'GCAL_IR_ON', 'LAMPON', 'GEMINI',
                                    'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20160724S0129.fits'): ['CAL', 'F2', 'LS', 'ARC', 'SPECT', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL',
                                    'UNPREPARED'],
    ('F2', 'S20160317S0053.fits'): ['ACQUISITION', 'F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20160317S0055.fits'): ['ACQUISITION', 'F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL',
                                    'UNPREPARED'],
    # for MASK: MASKNAME != 'None'

    #   F2 MOS
    ('F2', 'S20120104S0072.fits'): ['F2', 'MOS', 'SPECT', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20120104S0070.fits'): ['CAL', 'F2', 'MOS', 'FLAT', 'GCALFLAT', 'LAMPOFF', 'SPECT', 'GEMINI', 'SOUTH', 'RAW',
                                    'SIDEREAL', 'UNPREPARED', 'GCAL_IR_OFF'],
    ('F2', 'S20120104S0068.fits'): ['CAL', 'F2', 'MOS', 'ARC', 'SPECT', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL',
                                    'UNPREPARED'],
    ('F2', 'S20120104S0053.fits'): ['F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    # But the headers are crap, so OBSCLASS not set to "acq", so
    # won't work.  Needs to be replaced when really data available.
    # Note that old system does not define F2 ACQUISITION
    ('F2', 'S20120104S0055.fits'): ['F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    # For MASK, I think that MASKTYPE != 0 will do.
    # Same as above regarding ACQUISITION

    # Scientifically not really MOS, but technically should be the same
    # The headers are more likely to match with post-commissioning setup.
    ('F2', 'S20150622S0003.fits'): ['ACQUISITION', 'F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    # Should be able to use OBSCLASS to set ACQUISITION.
    ('F2', 'S20150622S0004.fits'): ['ACQUISITION', 'F2', 'IMAGE', 'GEMINI', 'SOUTH', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('F2', 'S20150622S0005.fits'): ['F2', 'MOS', 'SPECT', 'GEMINI', 'SOUTH', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('F2', 'S20150622S0008.fits'): ['CAL', 'F2', 'MOS', 'ARC', 'SPECT', 'GEMINI', 'SOUTH', 'NON_SIDEREAL', 'RAW',
                                    'UNPREPARED'],
    ('F2', 'S20150624S0172.fits'): ['CAL', 'F2', 'MOS', 'FLAT', 'GCALFLAT', 'LAMPON', 'GCAL_IR_ON', 'SPECT', 'GEMINI',
                                    'SOUTH', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],

    # GSAOI Data
    #   GSAOI Darks
    ('GSAOI', 'S20160422S0150.fits'): ['AT_ZENITH', 'AZEL_TARGET', 'CAL', 'GEMINI', 'SOUTH', 'GSAOI', 'DARK',
                                       'NON_SIDEREAL', 'RAW', 'UNPREPARED'],

    #   GSAOI Imaging
    ('GSAOI', 'S20150110S0142.fits'): ['GEMINI', 'SOUTH', 'GSAOI', 'IMAGE', 'RAW', 'SIDEREAL', 'UNPREPARED'],
    ('GSAOI', 'S20150105S0206.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'SOUTH', 'GSAOI', 'IMAGE', 'FLAT', 'DOMEFLAT',
                                       'LAMPOFF', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('GSAOI', 'S20150105S0167.fits'): ['AZEL_TARGET', 'CAL', 'GEMINI', 'SOUTH', 'GSAOI', 'IMAGE', 'FLAT', 'DOMEFLAT',
                                       'LAMPON', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    ('GSAOI', 'S20130524S0081_flat.fits'): ['CAL', 'GEMINI', 'SOUTH', 'GSAOI', 'IMAGE', 'FLAT', 'DOMEFLAT',
                                            'NON_SIDEREAL', 'PREPARED', 'PROCESSED', 'LAMPON'],
    # This is a processed image, so we don't care about LAMPON, but it's harmless, and easier to just keep it there

    # GHOST bundle and extracted files
    ('GHOST', 'S20230214S0025.fits'): ['UNPREPARED', 'SOUTH', 'RAW', 'GHOST', 'GEMINI', 'SIDEREAL', 'BUNDLE'],
    ('GHOST', 'S20230214S0025_1x4_blue001.fits'): ['BLUE', 'GHOST', 'SIDEREAL', 'UNPREPARED', 'SPECT', 'SOUTH', 'RAW', 'GEMINI', 'XD'],
    ('GHOST', 'S20230214S0025_1x4_red002.fits'): ['RED', 'GHOST', 'SIDEREAL', 'UNPREPARED', 'SPECT', 'SOUTH', 'RAW', 'GEMINI', 'XD'],
    ('GHOST', 'S20230214S0025_2x2_slit.fits'): ['SLIT', 'GHOST', 'SIDEREAL', 'SLITV', 'UNPREPARED', 'SOUTH', 'IMAGE', 'RAW', 'GEMINI'],

    #    ('bhros', 'S20070131S0030.fits') : ['BHROS', 'SIDEREAL', 'GEMINI_SOUTH', 'SPECT', 'UNPREPARED', 'RAW'],
    #    ('bhros', 'S20070131S0152.fits') : ['AT_ZENITH', 'BHROS', 'AZEL_TARGET', 'NON_SIDEREAL', 'GEMINI_SOUTH', 'SPECT', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20150629S0208.fits') : ['MOS', 'SPECT', 'F2', 'NON_SIDEREAL', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20150629S0245.fits') : ['FLAT', 'AT_ZENITH', 'GCAL_IR_ON', 'AZEL_TARGET', 'MOS', 'SPECT', 'F2', 'CAL', 'NON_SIDEREAL', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20150629S0244.fits') : ['AT_ZENITH', 'ARC', 'AZEL_TARGET', 'MOS', 'SPECT', 'F2', 'CAL', 'NON_SIDEREAL', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20160616S0072.fits') : ['GCAL_IR_ON', 'FLAT', 'SIDEREAL', 'SPECT', 'F2', 'CAL', 'LS', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20160616S0075.fits') : ['IMAGE', 'SIDEREAL', 'F2', 'ACQUISITION', 'IMAGE', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20160708S0025.fits') : ['AT_ZENITH', 'AZEL_TARGET', 'DARK', 'F2', 'CAL', 'NON_SIDEREAL', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('F2', 'S20160616S0071.fits') : ['ARC', 'SIDEREAL', 'SPECT', 'F2', 'CAL', 'LS', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('gmos', 'S20051225S0084.fits') : ['GMOS', 'IFU', 'BLUE', 'SIDEREAL', 'SPECT', 'GEMINI_SOUTH', 'RAW', 'UNPREPARED'],
    #    ('gmos', 'N20160616S0431.fits') : ['GMOS', 'ARC', 'GEMINI_NORTH', 'LS', 'AZEL_TARGET', 'CAL', 'SPECT', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    #    ('gmos', 'N20160516S0485.fits') : ['GMOS', 'AT_ZENITH', 'GEMINI_NORTH', 'BIAS', 'AZEL_TARGET', 'CAL', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    #    ('gmos', 'N20160407S0076.fits') : ['GMOS', 'GEMINI_NORTH', 'BIAS', 'AZEL_TARGET', 'CAL', 'NON_SIDEREAL', 'RAW', 'UNPREPARED'],
    #    ('gmos', 'N20111124S0203.fits') : ['GMOS', 'IFU', 'RED', 'GEMINI_NORTH', 'FLAT', 'CAL', 'SIDEREAL', 'SPECT', 'RAW', 'UNPREPARED'],
    #    ('gmos', 'N20160516S0485_bias.fits') : ['GMOS', 'AT_ZENITH', 'GEMINI_NORTH', 'BIAS', 'AZEL_TARGET', 'CAL', 'NON_SIDEREAL', 'PREPARED', 'PROCESSED_BIAS'],
    #    ('gmos', 'S20120815S0031.fits') : ['GMOS', 'IMAGE', 'SIDEREAL', 'IMAGE', 'GEMINI_SOUTH', 'RAW', 'UNPREPARED'],
    #    ('GNIRS', 'N20160523S0191.fits'): ['GNIRS', 'GEMINI_NORTH', 'SIDEREAL', 'LS', 'SPECT', 'UNPREPARED', 'RAW'],
    #    ('gpi', 'S20160523S0215.fits') : ['GPI', 'IFU', 'NON_SIDEREAL', 'SPECT', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('GSAOI', 'S20150531S0009.fits') : ['FLAT', 'TWILIGHT', 'SIDEREAL', 'CAL', 'IMAGE', 'GSAOI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('GSAOI', 'S20160422S0150.fits') : ['AT_ZENITH', 'AZEL_TARGET', 'DARK', 'CAL', 'IMAGE', 'NON_SIDEREAL', 'GSAOI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('GSAOI', 'S20160422S0092.fits') : ['SIDEREAL', 'IMAGE', 'GSAOI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('GSAOI', 'S20160613S0316.fits') : ['DOMEFLAT', 'FLAT', 'AZEL_TARGET', 'CAL', 'IMAGE', 'NON_SIDEREAL', 'GSAOI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('michelle', 'N20030625S0075.fits') : ['GEMINI_NORTH', 'MICHELLE', 'SIDEREAL', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20130715S0112.fits') : ['IMAGE', 'SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20130607S0592.fits') : ['IMAGE', 'AZEL_TARGET', 'DARK', 'CAL', 'NON_SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20130714S0155.fits') : ['CAL', 'FLAT', 'IMAGE', 'GCAL_IR_ON', 'SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20130607S0258.fits') : ['ADI_R', 'IMAGE', 'SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20091206S0101.fits') : ['AT_ZENITH', 'CAL', 'FLAT', 'IMAGE', 'AZEL_TARGET', 'NON_SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('nici', 'S20130806S0071.fits') : ['ASDI', 'IMAGE', 'SIDEREAL', 'NICI', 'GEMINI_SOUTH', 'UNPREPARED', 'RAW'],
    #    ('NIFS', 'N20160428S0174.fits') : ['NIFS', 'IFU', 'GEMINI_NORTH', 'AZEL_TARGET', 'NON_SIDEREAL', 'SPECT', 'UNPREPARED', 'RAW'],
    #    ('NIRI', 'N20160523S0471_forStack.fits') : ['NIRI', 'GEMINI_NORTH', 'SIDEREAL', 'IMAGE', 'PREPARED'],
    #    ('NIRI', 'N20120505S0564.fits') : ['NIRI', 'GEMINI_NORTH', 'AZEL_TARGET', 'SPECT', 'LS', 'NON_SIDEREAL', 'GCAL_IR_OFF', 'UNPREPARED', 'RAW'],
    #    ('NIRI', 'N20160522S0004.fits') : ['NIRI', 'GEMINI_NORTH', 'SIDEREAL', 'IMAGE', 'UNPREPARED', 'RAW'],
}
