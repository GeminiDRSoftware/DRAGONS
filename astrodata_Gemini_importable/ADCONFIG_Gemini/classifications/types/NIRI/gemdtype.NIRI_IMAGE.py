
class NIRI_IMAGE(DataClassification):
    name="NIRI_IMAGE"
    usage = "Applies to any IMAGE dataset from the NIRI instrument."
    parent = "NIRI"
    requirement = ISCLASS('NIRI') & PHU({"{prohibit}FILTER3":"(.*?)grism(.*?)"}) & NOT(ISCLASS("NIRI_DARK"))


newtypes.append(NIRI_IMAGE())
