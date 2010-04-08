class IMAGE(DataClassification):
    name="IMAGE"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Applies to all Gemini imaging datasets.
        '''
    parent = "GENERIC"
    requirement = ISCLASS("GMOS_IMAGE") | ISCLASS("NIRI_IMAGE")
    
newtypes.append( IMAGE())
