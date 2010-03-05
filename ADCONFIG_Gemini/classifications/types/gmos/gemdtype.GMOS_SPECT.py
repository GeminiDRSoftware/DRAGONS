class GMOS_SPECT(DataClassification):
    name="GMOS_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    typeReqs= ['GMOS']
    phuReqs = {
                # If the grating is not the mirror, then it must be spectroscopy
                '{prohibit}GRATING': 'MIRROR',
                '{prohibit}OBSTYPE': 'BIAS'
                }

newtypes.append( GMOS_SPECT())
