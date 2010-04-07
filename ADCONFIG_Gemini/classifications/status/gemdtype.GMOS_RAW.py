class GMOS_RAW(DataClassification):
    editprotect=True
    name="GMOS_RAW"
    usage = 'Applies to RAW GMOS data.'
    typeReqs= ['GMOS']
    requirement = ISCLASS("RAW") & ISCLASS("GMOS")
    
newtypes.append(GMOS_RAW())
