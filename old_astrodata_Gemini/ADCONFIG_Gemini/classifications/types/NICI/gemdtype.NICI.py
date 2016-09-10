class NICI(DataClassification):
    name="NICI"
    usage = "Applies to all datasets taken with the NICI instrument."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='NICI')

newtypes.append(NICI())
