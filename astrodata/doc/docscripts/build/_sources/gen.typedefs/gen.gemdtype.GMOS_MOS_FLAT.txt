
GMOS_MOS_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_MOS_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_MOS_FLAT.py

.. code-block:: python
    :linenos:

    class GMOS_MOS_FLAT(DataClassification):
        name="GMOS_MOS_FLAT"
        usage = """
            Applies to all MOS flat datasets from the GMOS instruments
            """
        parent = "GMOS_MOS"
        requirement = AND([  ISCLASS("GMOS_MOS"),
                             PHU(OBSTYPE="FLAT"),
                             NOT(ISCLASS("GMOS_MOS_TWILIGHT"))  ])
    
    newtypes.append(GMOS_MOS_FLAT())



