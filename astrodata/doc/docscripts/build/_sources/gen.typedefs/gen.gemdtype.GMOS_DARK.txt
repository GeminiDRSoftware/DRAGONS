
GMOS_DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_DARK

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_DARK.py

.. code-block:: python
    :linenos:

    class GMOS_DARK(DataClassification):
        name="GMOS_DARK"
        usage = """
            Applies to all dark datasets from the GMOS instruments
            """
        parent = "GMOS"
        requirement = ISCLASS("GMOS") & PHU(OBSTYPE="DARK")
    
    newtypes.append(GMOS_DARK())



