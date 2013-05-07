
GSAOI_DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GSAOI_DARK

Source Location 
    ADCONFIG_Gemini/classifications/types/GSAOI/gemdtype.GSAOI_DARK.py

.. code-block:: python
    :linenos:

    class GSAOI_DARK(DataClassification):
        name="GSAOI_DARK"
        usage = """
            Applies to all dark datasets from the GSAOI instrument
            """
        parent = "GSAOI"
        requirement = ISCLASS("GSAOI") & PHU(OBSTYPE="DARK")
    
    newtypes.append(GSAOI_DARK())



