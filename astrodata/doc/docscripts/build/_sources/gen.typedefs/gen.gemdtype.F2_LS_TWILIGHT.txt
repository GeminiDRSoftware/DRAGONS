
F2_LS_TWILIGHT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_LS_TWILIGHT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_LS_TWILIGHT.py

.. code-block:: python
    :linenos:

    class F2_LS_TWILIGHT(DataClassification):
        name="F2_LS_TWILIGHT"
        usage = """
            Applies to all longslit twilight flat datasets from the FLAMINGOS-2
            instrument
            """
        parent = "F2_LS"
        requirement = AND([  ISCLASS("F2_LS"),
                             PHU(OBSTYPE="FLAT"),
                             PHU(OBJECT="Twilight")  ])
    
    newtypes.append(F2_LS_TWILIGHT())



