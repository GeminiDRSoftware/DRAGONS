class FRINGE(DataClassification):
    name="FRINGE"
    usage = "A processed fringe."
    parent = "CAL"
    requirement = PHU(GIFRINGE='(.*?)')

newtypes.append(FRINGE())
