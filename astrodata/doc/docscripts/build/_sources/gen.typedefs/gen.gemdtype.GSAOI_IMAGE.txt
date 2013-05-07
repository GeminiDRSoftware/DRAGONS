
GSAOI_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GSAOI_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/GSAOI/gemdtype.GSAOI_IMAGE.py

.. code-block:: python
    :linenos:

    class GSAOI_IMAGE(DataClassification):
        name="GSAOI_IMAGE"
        usage = """
            Applies to all imaging datasets from the GSAOI instrument
            """
        parent = "GSAOI"
        requirement = ISCLASS("GSAOI")
    
    newtypes.append(GSAOI_IMAGE())



