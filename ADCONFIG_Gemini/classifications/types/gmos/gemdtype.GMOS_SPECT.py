class GMOS_SPECT(DataClassification):
    name="GMOS_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    parent = "GMOS"
    requirement = AND(ISCLASS('GMOS'),
                      PHU({'{prohibit}GRATING': 'MIRROR'}),
                      NOT(ISCLASS("GMOS_BIAS")))

newtypes.append( GMOS_SPECT())
