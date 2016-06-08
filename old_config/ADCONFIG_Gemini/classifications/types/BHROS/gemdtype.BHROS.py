class BHROS(DataClassification):
    name="BHROS"
    usage = ""
    typeReqs= []
    phuReqs= {}
    parent = "GEMINI"
    requirement = PHU(INSTRUME='bHROS')

newtypes.append(BHROS())
