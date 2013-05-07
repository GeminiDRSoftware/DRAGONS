
GEMINI_SOUTH Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GEMINI_SOUTH

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.GEMINI_SOUTH.py

.. code-block:: python
    :linenos:

    
    class GEMINI_SOUTH(DataClassification):
        name="GEMINI_SOUTH"
        usage = "Applies to datasets from instruments at Gemini South."
        
        parent = "GEMINI"
        requirement = PHU(TELESCOP='Gemini-South',OBSERVAT='Gemini-South')
    
    newtypes.append(GEMINI_SOUTH())



