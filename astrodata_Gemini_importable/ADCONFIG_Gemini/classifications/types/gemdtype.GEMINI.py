class GEMINI(DataClassification):
    name="GEMINI"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
        '''
    # Added the instrument names directly, so that when we get engineering data that does
    # not have telescope headers in, and thus doesn't identify as GEMINI_NORTH or _SOUTH
    # then it does identify as GEMINI, so that the gemini descriptors associate with it.
    requirement = OR(ISCLASS("GEMINI_NORTH"),
                     ISCLASS("GEMINI_SOUTH"),
                     ISCLASS("GMOS"),
                     ISCLASS("NIRI"),
                     ISCLASS("GNIRS"),
                     ISCLASS("MICHELLE"),
                     ISCLASS("NICI"),
                     ISCLASS("F2"),
                     ISCLASS("NIFS"),
                     ISCLASS("TRECS"),
                     ISCLASS("GSAOI"),
                     ISCLASS("BHROS"))
               

newtypes.append( GEMINI())
