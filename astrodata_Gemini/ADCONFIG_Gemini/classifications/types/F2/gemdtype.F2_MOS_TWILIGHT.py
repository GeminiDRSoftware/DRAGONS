class F2_MOS_TWILIGHT(DataClassification):
    name="F2_MOS_TWILIGHT"
    usage = """
        Applies to all MOS twilight flat datasets from the FLAMINGOS-2 
        instrument
        """
    parent = "F2_MOS"
    requirement = AND([  ISCLASS("F2_MOS"),
                         PHU(OBSTYPE="FLAT"),
                         PHU(OBJECT="Twilight")  ])

newtypes.append(F2_MOS_TWILIGHT())
