
class NIRI_SPECT(DataClassification):
    name="NIRI_SPECT"
    usage = ""
    requirement = ISCLASS('NIRI') & PHU(FILTER3='(.*?)grism(.*?)')

newtypes.append(NIRI_SPECT())
