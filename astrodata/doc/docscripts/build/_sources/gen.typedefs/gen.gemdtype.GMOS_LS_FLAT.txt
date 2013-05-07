
GMOS_LS_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_LS_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_LS_FLAT.py

.. code-block:: python
    :linenos:

    class GMOS_LS_FLAT(DataClassification):
        name="GMOS_LS_FLAT"
        usage = """
            Applies to all longslit flat datasets from the GMOS instruments
            """
        parent = "GMOS_LS"
        requirement = AND([  ISCLASS("GMOS_LS"),
                             PHU(OBSTYPE="FLAT"),
                             NOT(ISCLASS("GMOS_LS_TWILIGHT"))  ])
    
    newtypes.append(GMOS_LS_FLAT())



