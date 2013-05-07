
MICHELLE_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    MICHELLE_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/MICHELLE/gemdtype.MICHELLE_IMAGE.py

.. code-block:: python
    :linenos:

    class MICHELLE_IMAGE(DataClassification):
        name = "MICHELLE_IMAGE"
        usage = "Applies to all imaging datasets from the MICHELLE instrument"
        parent = "MICHELLE"
        requirement = ISCLASS("MICHELLE") & PHU(CAMERA="imaging")
    
    newtypes.append(MICHELLE_IMAGE())



