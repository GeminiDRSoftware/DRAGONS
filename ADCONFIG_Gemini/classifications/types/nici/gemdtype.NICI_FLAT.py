class NICI_FLAT(DataClassification):
    name="NICI_FLAT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all NICI flats 
        '''
    parent = "NICI_CAL"
    requirement = ISCLASS('NICI') & PHU(OBSTYPE='FLAT')

newtypes.append( NICI_FLAT())
