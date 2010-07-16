class SPECT(DataClassification):
    name="SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all Gemini spectroscopy datasets.
        '''
    parent = "GENERIC"
    requirement = OR([  ISCLASS("GMOS_SPECT"),
                        ISCLASS("NIRI_SPECT"),
                        ISCLASS("MICHELLE_SPECT"),
                        ISCLASS("TRECS_SPECT"),
                        ISCLASS("GNIRS_SPECT"),
                        ISCLASS("NIFS_SPECT")
                    ])
                        
    
newtypes.append( SPECT())
