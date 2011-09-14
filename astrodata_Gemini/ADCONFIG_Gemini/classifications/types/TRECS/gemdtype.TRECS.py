class TRECS(DataClassification):
    name = "TRECS"
    usage = "Applies to all datasets from the TRECS instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME="TReCS")

newtypes.append(TRECS())
