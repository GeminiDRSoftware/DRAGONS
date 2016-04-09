
class GMOS_NODANDSHUFFLE(DataClassification):
    name="GMOS_NODANDSHUFFLE"
    usage = """Applies to N&S GMOS_SPECT datasets"""
    typeReqs= []
    phuReqs= {}
    parent = "GMOS_SPECT"
    requirement = AND ([ ISCLASS("GMOS_SPECT"),
                         PHU(NODPIX='.*') ])

newtypes.append(GMOS_NODANDSHUFFLE())
