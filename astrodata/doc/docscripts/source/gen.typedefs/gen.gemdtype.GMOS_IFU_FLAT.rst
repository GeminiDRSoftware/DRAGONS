
GMOS_IFU_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IFU_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IFU_FLAT.py

.. code-block:: python
    :linenos:

    class GMOS_IFU_FLAT(DataClassification):
        name="GMOS_IFU_FLAT"
        usage = """
            Applies to all IFU flat datasets from the GMOS instruments
            """
        parent = "GMOS_IFU"
        requirement = AND([  ISCLASS("GMOS_IFU"),
                             PHU(OBSTYPE="FLAT"),
                             NOT(ISCLASS("GMOS_IFU_TWILIGHT"))  ])
    
    newtypes.append(GMOS_IFU_FLAT())



