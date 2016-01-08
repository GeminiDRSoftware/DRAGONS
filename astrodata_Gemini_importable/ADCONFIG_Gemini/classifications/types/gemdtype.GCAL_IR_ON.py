class GCAL_IR_ON(DataClassification):
    name="GCAL_IR_ON"
    usage = "Indicates that the GCAL IR flat field lamp is on and the shutter is open. This is typically referred to as a lamp-on flat"
    
    parent = "GEMINI"
    requirement = PHU(GCALLAMP='IRhigh') & PHU(GCALSHUT='OPEN')

newtypes.append(GCAL_IR_ON())
