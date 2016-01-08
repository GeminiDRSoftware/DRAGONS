class GENERIC(DataClassification):
    name="GENERIC"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Special parent to group generic types (e.g. IMAGE, SPECT, MOS, IFU)
        """
    parent = None
    requirement = False # no type is "GENERIC"

newtypes.append(GENERIC())
