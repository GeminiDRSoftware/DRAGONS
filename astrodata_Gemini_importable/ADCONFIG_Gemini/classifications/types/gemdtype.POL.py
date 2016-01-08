class POL(DataClassification):
    name="POL"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all Gemini polarimetry datasets
        """
    parent = "GENERIC"
    requirement = OR([  ISCLASS("GPI_POL")])

newtypes.append(POL())
