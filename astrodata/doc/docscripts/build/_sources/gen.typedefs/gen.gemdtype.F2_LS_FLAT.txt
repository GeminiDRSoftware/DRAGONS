
F2_LS_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_LS_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_LS_FLAT.py

.. code-block:: python
    :linenos:

    class F2_LS_FLAT(DataClassification):
        name="F2_LS_FLAT"
        usage = """
            Applies to all longslit flat datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2_LS"
        requirement = AND([  ISCLASS("F2_LS"),
                             PHU(OBSTYPE="FLAT"),
                             NOT(ISCLASS("F2_LS_TWILIGHT"))  ])
    
    newtypes.append(F2_LS_FLAT())



