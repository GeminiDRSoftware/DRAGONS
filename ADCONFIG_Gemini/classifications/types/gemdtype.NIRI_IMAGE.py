
class NIRI_IMAGE(DataClassification):
    name="NIRI_IMAGE"
    usage = ""
    typeReqs= ['NIRI']
    phuReqs= {'FILTER3': '(.*?)Pupil(.*?)'}

newtypes.append(NIRI_IMAGE())
