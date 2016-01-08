
class NICI_ASDI(DataClassification):
    name="NICI_ASDI"
    usage = "Applies to imaging datasets from the NICI instrument."
    parent = "NICI"
    # DICHROIC PHU keyword value contains the string '50/50'
    requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?50/50.*?" }) & \
                  PHU(CRMODE='FIXED') & PHU({'{prohibit}OBSTYPE': 'FLAT'})

newtypes.append(NICI_ASDI())
