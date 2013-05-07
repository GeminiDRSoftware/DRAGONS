
GEMINI_NORTH Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GEMINI_NORTH

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.GEMINI_NORTH.py

.. code-block:: python
    :linenos:

    
    class GEMINI_NORTH(DataClassification):
        name="GEMINI_NORTH"
        usage = "Data taken at Gemini North upon Mauna Kea."
        
        parent = "GEMINI"
        requirement = PHU({'TELESCOP': 'Gemini-North', 'OBSERVAT': 'Gemini-North'})
    
    newtypes.append(GEMINI_NORTH())



