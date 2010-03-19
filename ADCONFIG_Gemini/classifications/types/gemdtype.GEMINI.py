class GEMINI(DataClassification):
    name="GEMINI"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    requirement = ISCLASS("GEMINI_NORTH", "GEMINI_SOUTH")

newtypes.append( GEMINI())
