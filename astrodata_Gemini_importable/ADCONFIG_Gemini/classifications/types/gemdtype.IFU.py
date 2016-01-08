class IFU(DataClassification):
    name="IFU"
    usage = """
        Applies to all Gemini integral field unit spectroscopy datasets
        """
    parent = "SPECT"
    requirement = OR([  ISCLASS("GMOS_IFU"),
                        ISCLASS("NIFS_SPECT"),
                        ISCLASS("GPI_SPECT"),
                        ISCLASS("GNIRS_IFU")])

newtypes.append(IFU())
