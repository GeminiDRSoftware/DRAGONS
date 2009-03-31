class GMOS(ORClassification):
    name="GMOS"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    typeORs = ["GMOS_N", "GMOS_S"]

newtypes.append( GMOS())
