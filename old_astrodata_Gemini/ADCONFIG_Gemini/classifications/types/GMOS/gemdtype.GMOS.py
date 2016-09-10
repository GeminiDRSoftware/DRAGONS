class GMOS(DataClassification):
    name="GMOS"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
        
    parent = "GEMINI"
    requirement = ISCLASS("GMOS_N") | ISCLASS("GMOS_S") | PHU(INSTRUME='GMOS')
    # equivalent to...
    #requirement = OR(   
    #                    ClassReq("GMOS_N"), 
    #                    ClassReq("GMOS_S")
    #                    )

newtypes.append( GMOS())
