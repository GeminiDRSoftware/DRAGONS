class MICHELLE_SPECT(DataClassification):
    name = "MICHELLE_SPECT"
    usage = """
        Applies to all spectroscopic datasets from the MICHELLE instrument
        """
    parent = "MICHELLE"
    requirement = ISCLASS("MICHELLE") & PHU(CAMERA="spectroscopy")

newtypes.append(MICHELLE_SPECT())
