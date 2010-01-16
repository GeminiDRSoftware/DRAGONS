
class NIRI_SPECT(DataClassification):
    name="NIRI_SPECT"
    usage = ""
    typeReqs= ['NIRI']
    phuReqs= {'FILTER3': '(.*?)grism(.*?)'}

newtypes.append(NIRI_SPECT())
