
class GSAOI(DataClassification):
    name="GSAOI"
    usage = "Applies to any data from the GSAOI instrument."
    parent = "GEMINI"

    requirement = PHU(INSTRUME='GSAOI')

newtypes.append(GSAOI())
