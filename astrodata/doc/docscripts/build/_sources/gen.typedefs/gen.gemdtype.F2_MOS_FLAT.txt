
F2_MOS_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_MOS_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_MOS_FLAT.py

.. code-block:: python
    :linenos:

    class F2_MOS_FLAT(DataClassification):
        name="F2_MOS_FLAT"
        usage = """
            Applies to all MOS flat datasets from the FLAMINGOS-2 instrument
            """
        parent = "F2_MOS"
        requirement = AND([  ISCLASS("F2_MOS"),
                             PHU(OBSTYPE="FLAT"),
                             NOT(ISCLASS("F2_MOS_TWILIGHT"))  ])
    
    newtypes.append(F2_MOS_FLAT())



