
class NIFS_IMAGE(DataClassification):
    name="NIFS_IMAGE"
    usage = "Applies to any image dataset from the NIFS instrument."
    parent = "NIFS"

    requirement = ISCLASS("NIFS") & PHU( FILTER3='(.*?)pupil(.*?)')

newtypes.append(NIFS_IMAGE())
