
MICHELLE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    MICHELLE

Source Location 
    ADCONFIG_Gemini/classifications/types/MICHELLE/gemdtype.MICHELLE.py

.. code-block:: python
    :linenos:

    class MICHELLE(DataClassification):
        name = "MICHELLE"
        usage = "Applies to all datasets from the MICHELLE instrument"
        parent = "GEMINI"
        requirement = PHU(INSTRUME="michelle")
    
    newtypes.append(MICHELLE())



