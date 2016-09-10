class GPI_POL(DataClassification):
    name="GPI_POL"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all polarimetry datasets from the GPI instrument
        """
    parent = "GPI"
    requirement = AND([  ISCLASS("GPI"),
                         (PHU(DISPERSR="DISP_WOLLASTON*"))])

newtypes.append(GPI_POL())
