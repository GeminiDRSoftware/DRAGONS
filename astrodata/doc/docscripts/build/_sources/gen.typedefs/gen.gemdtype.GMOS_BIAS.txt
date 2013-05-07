
GMOS_BIAS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_BIAS

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_BIAS.py

.. code-block:: python
    :linenos:

    class GMOS_BIAS(DataClassification):
        name="GMOS_BIAS"
        usage = """
            Applies to all bias datasets from the GMOS instruments
            """
        parent = "GMOS"
        requirement = ISCLASS("GMOS") & PHU(OBSTYPE="BIAS")
    
    newtypes.append(GMOS_BIAS())



