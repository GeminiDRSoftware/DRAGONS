
class F2(DataClassification):
    name="F2"
    usage = "Applies to all datasets from the F2 instrument."
    parent = "GEMINI"
    # Early commissioning from ~2010-01/02 data used 'Flam' as the instrume header
    # The final string is TBD, but we'll assume Flamingos2 for now
    requirement = PHU(INSTRUME='Flam') | PHU(INSTRUME='Flamingos2')

newtypes.append(F2())
