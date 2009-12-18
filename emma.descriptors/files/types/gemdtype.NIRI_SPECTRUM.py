
class NIRI_SPECTRUM(DataClassification):
    name="NIRI_SPECTRUM"
    usage = ""
    typeReqs= ['NIRI']
    phuReqs= {'FILTER3': '(.*?)grism(.*?)'}

newtypes.append(NIRI_SPECTRUM())
