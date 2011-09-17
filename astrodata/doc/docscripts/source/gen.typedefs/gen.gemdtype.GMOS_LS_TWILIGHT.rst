
GMOS_LS_TWILIGHT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_LS_TWILIGHT

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_LS_TWILIGHT.py

.. code-block:: python
    :linenos:

    class GMOS_LS_TWILIGHT(DataClassification):
        name="GMOS_LS_TWILIGHT"
        usage = """
            Applies to all longslit twilight flat datasets from the GMOS
            instruments
            """
        parent = "GMOS_LS"
        requirement = AND([  ISCLASS("GMOS_LS"),
                             PHU(OBSTYPE="FLAT"),
                             PHU(OBJECT="Twilight")  ])
    
    newtypes.append(GMOS_LS_TWILIGHT())



