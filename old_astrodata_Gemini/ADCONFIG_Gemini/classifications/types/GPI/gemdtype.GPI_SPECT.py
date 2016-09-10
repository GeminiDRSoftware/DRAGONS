class GPI_SPECT(DataClassification):
    name="GPI_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all spectroscopic datasets from the GPI instrument
        """
    parent = "GPI"
    requirement = AND([  ISCLASS("GPI"),
                         (PHU(DISPERSR="DISP_PRISM*"))])

newtypes.append(GPI_SPECT())
