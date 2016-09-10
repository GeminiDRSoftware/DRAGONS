class MICHELLE_IMAGE(DataClassification):
    name = "MICHELLE_IMAGE"
    usage = "Applies to all imaging datasets from the MICHELLE instrument"
    parent = "MICHELLE"
    requirement = ISCLASS("MICHELLE") & PHU(CAMERA="imaging")

newtypes.append(MICHELLE_IMAGE())
