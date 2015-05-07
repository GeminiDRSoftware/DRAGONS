
class GNIRS(DataClassification):
    name="GNIRS"
    usage = "Applies to all datasets from the GNIRS instrument."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='GNIRS') | PHU(INSTRUME='gnirs')

newtypes.append(GNIRS())
