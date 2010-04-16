
class NICI_ADI_B(DataClassification):
    name="NICI_ADI_B"
    usage = "Applies to imaging datasets from the NICI instrument."
    parent = "NICI"
    # DICHROIC PHU keyword value contains the string 'Mirror'
    requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?Mirror*?" })

newtypes.append(NICI_ADI_B())
