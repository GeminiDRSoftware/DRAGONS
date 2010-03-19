class FRINGE(DataClassification):
    name="FRINGE"
    usage = "A processed fringe."
    requirement = PHU(GIFRINGE='(.*?)')

newtypes.append(FRINGE())
