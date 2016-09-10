class GCAL_IR_OFF(DataClassification):
    name="GCAL_IR_OFF"
    usage = "Indicates that the GCAL IR flat field lamp is on and the shutter is closed. This is typically referred to as a lamp-off flat. The subelty that the lamp is actually on, but the shutter is closed is lost on most people and abstracted away here. :-)"
    
    parent = "GEMINI"
    requirement = PHU(GCALLAMP='IRhigh') & PHU(GCALSHUT='CLOSED')

newtypes.append(GCAL_IR_OFF())
