
class NIRI_IMAGE(DataClassification):
    name="NIRI_IMAGE"
    usage = "Applies to any IMAGE dataset from the NIRI instrument."
    parent = "NIRI"
    requirement = ISCLASS('NIRI') & PHU(FILTER3='(.*?)[Pp]upil(.*?)')

newtypes.append(NIRI_IMAGE())
