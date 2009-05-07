class GMOS_SPECT(ORClassification):
    name="GMOS_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    typeORs = ["GMOS_IFU", "GMOS_MOS", "GMOS_LS"]

newtypes.append( GMOS_SPECT())
