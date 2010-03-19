
class NIFS_IMAGE(DataClassification):
    name="NIFS_IMAGE"
    usage = ""
    requirement = ISCLASS("NIFS") & PHU( FILTER3='(.*?)pupil(.*?)')

newtypes.append(NIFS_IMAGE())
