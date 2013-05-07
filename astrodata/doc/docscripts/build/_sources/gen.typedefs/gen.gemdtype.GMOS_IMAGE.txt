
GMOS_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS_IMAGE.py

.. code-block:: python
    :linenos:

    class GMOS_IMAGE(DataClassification):
        name="GMOS_IMAGE"
        usage = """
            Applies to all imaging datasets from the GMOS instruments
            """
        parent = "GMOS"
        requirement = AND([  ISCLASS("GMOS"),
                             PHU(GRATING="MIRROR"),
                             NOT(ISCLASS("GMOS_BIAS"))  ])
    
    newtypes.append(GMOS_IMAGE())



