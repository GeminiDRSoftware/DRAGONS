class IMAGE(DataClassification):
    name="IMAGE"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Special parent to group generic types (e.g. IMAGE, SPECT, MOS, IFU)
        '''
    parent = "GENERIC"
    requirement = ISCLASS("GMOS_IMAGE") | ISCLASS("NIRI_IMAGE")
    
newtypes.append( IMAGE())
