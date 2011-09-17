
GMOS_LS_ARC Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_LS_ARC

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_LS_ARC.py

.. code-block:: python
    :linenos:

    class GMOS_LS_ARC(DataClassification):
        name="GMOS_LS_ARC"
        usage = """
            Applies to all longslit arc datasets from the GMOS instruments
            """
        parent = "GMOS_LS"
        requirement = AND([  ISCLASS("GMOS_LS"),
                             PHU(OBSTYPE="ARC")  ])
    
    newtypes.append(GMOS_LS_ARC())



