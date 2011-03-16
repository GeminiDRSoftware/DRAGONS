
class NIRI_SPECT(DataClassification):
    name="NIRI_SPECT"
    usage = "Applies to any spectra from the NIRI instrument."
    parent = "NIRI"
    requirement = ISCLASS('NIRI') & PHU(FILTER3='(.*?)grism(.*?)')

newtypes.append(NIRI_SPECT())
