
GMOS_MOS_TWILIGHT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_MOS_TWILIGHT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_MOS_TWILIGHT.py

.. code-block:: python
    :linenos:

    class GMOS_MOS_TWILIGHT(DataClassification):
        name="GMOS_MOS_TWILIGHT"
        usage = """
            Applies to all MOS twilight flat datasets from the GMOS instruments
            """
        parent = "GMOS_MOS"
        requirement = AND([  ISCLASS("GMOS_MOS"),
                             PHU(OBSTYPE="FLAT"),
                             PHU(OBJECT="Twilight")  ])
    
    newtypes.append(GMOS_MOS_TWILIGHT())



