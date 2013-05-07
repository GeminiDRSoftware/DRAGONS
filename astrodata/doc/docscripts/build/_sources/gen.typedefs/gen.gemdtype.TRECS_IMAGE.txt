
TRECS_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    TRECS_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/TRECS/gemdtype.TRECS_IMAGE.py

.. code-block:: python
    :linenos:

    class TRECS_IMAGE(DataClassification):
        name = "TRECS_IMAGE"
        usage = "Applies to all imaging datasets from the TRECS instrument"
        parent = "TRECS"
        requirement = ISCLASS("TRECS") & PHU(GRATING="(.*?)[mM]irror")
    
    newtypes.append(TRECS_IMAGE())



