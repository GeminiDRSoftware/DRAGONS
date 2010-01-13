class NICI_FLAT(DataClassification):
    name="NICI_FLAT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all NICI data 
        '''
    typeReqs= ['NICI']
    phuReqs= {'OBSTYPE': 'FLAT'}

newtypes.append( NICI_FLAT())
