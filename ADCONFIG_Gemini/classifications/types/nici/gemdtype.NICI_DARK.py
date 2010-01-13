class NICI_DARK(ORClassification):
    name="NICI_DARK"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    typeORs = ["NICI_DARK_CURRENT", "NICI_DARK_OLD"]

newtypes.append( NICI_DARK())
