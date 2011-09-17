
GMOS_IFU_ARC Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IFU_ARC

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IFU_ARC.py

.. code-block:: python
    :linenos:

    class GMOS_IFU_ARC(DataClassification):
        name="GMOS_IFU_ARC"
        usage = """
            Applies to all IFU arc datasets from the GMOS instruments
            """
        parent = "GMOS_IFU"
        requirement = AND([  ISCLASS("GMOS_IFU"),
                             PHU(OBSTYPE="ARC")  ])
    
    newtypes.append(GMOS_IFU_ARC())



