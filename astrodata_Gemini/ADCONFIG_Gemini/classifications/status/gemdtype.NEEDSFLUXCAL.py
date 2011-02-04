class NEEDSFLUXCAL(DataClassification):
    editprotect=False
    name="NEEDSFLUXCAL"
    usage = 'Applies to data ready for flux calibration.'
    requirement = ISCLASS("IMAGE") & ISCLASS("PREPARED")
    
newtypes.append(NEEDSFLUXCAL())
