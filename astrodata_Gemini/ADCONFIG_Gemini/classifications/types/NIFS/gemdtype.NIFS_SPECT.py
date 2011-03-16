
class NIFS_SPECT(DataClassification):
    name="NIFS_SPECT"
    usage = "Applies to any spectroscopy dataset from the NIFS instrument."
    parent = "NIFS"

    requirement = ISCLASS("NIFS") & PHU( FLIP='Out')

newtypes.append(NIFS_SPECT())
