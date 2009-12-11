class FRINGE(DataClassification):
    name="FRINGE"
    usage = "A processed fringe."
    phuReqs= {'GIFRINGE': '(*?)'}

newtypes.append(FRINGE())