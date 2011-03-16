
class HOKUPAAQUIRC(DataClassification):
    name="HOKUPAAQUIRC"
    usage = "Applies to datasets from the HOKUPAA+QUIRC instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='Hokupaa\+QUIRC')

newtypes.append(HOKUPAAQUIRC())
