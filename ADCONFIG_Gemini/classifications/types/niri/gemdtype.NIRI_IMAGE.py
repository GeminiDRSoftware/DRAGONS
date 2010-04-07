
class NIRI_IMAGE(DataClassification):
    name="NIRI_IMAGE"
    usage = ""
    requirement = ISCLASS('NIRI') & PHU(FILTER3='(.*?)[Pp]upil(.*?)')

newtypes.append(NIRI_IMAGE())
