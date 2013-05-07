
NIRI_DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI_DARK

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI_DARK.py

.. code-block:: python
    :linenos:

    class NIRI_DARK(DataClassification):
        name="NIRI_DARK"
        usage = """
            Applies to all dark datasets from the NIRI instrument
            """
        parent = "NIRI"
        requirement = ISCLASS("NIRI") & PHU(OBSTYPE="DARK")
    
    newtypes.append(NIRI_DARK())



