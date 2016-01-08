
class HRWFS(DataClassification):
    name="HRWFS"
    usage = "Applies to all datasets from the HRWFS instrument."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='hrwfs')

newtypes.append(HRWFS())
