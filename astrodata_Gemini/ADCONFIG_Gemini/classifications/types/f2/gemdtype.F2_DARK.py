class F2_DARK(DataClassification):
    name="F2_DARK"
    usage = """
        Applies to all dark datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_IMAGE"
    requirement = AND([  ISCLASS("F2_IMAGE"),
                         PHU(OBSTYPE="DARK")  ])

newtypes.append(F2_DARK())
