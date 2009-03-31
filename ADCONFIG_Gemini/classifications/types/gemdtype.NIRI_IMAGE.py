
class NIRI_IMAGE(DataClassification):
    name="NIRI_IMAGE"
    usage = ""
    typeReqs= ['NIRI']
    phuReqs= {'FILTER3': '(.*?)pupil(.*?)'}

newtypes.append(NIRI_IMAGE())
