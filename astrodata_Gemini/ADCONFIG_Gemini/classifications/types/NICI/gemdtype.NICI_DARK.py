class NICI_DARK(DataClassification):
    name="NICI_DARK"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all dark current calibration datasets for NICI instrument.
        '''
    parent = "NICI_CAL"
    requirement = ISCLASS("NICI_DARK_CURRENT", "NICI_DARK_OLD")

newtypes.append( NICI_DARK())
