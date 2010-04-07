class CAL(DataClassification):
    name="CAL"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = '''
        Special parent to group generic types (e.g. IMAGE, SPECT, MOS, IFU)
        '''
    parent = "GENERIC"
    requirement = ISCLASS("GMOS_CAL")

newtypes.append( CAL())
