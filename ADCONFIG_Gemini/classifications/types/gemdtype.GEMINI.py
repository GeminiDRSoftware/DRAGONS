class GEMINI(ORClassification):
    name="GEMINI"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    typeORs = ["GEMINI_NORTH", "GEMINI_SOUTH"]

newtypes.append( GEMINI())
