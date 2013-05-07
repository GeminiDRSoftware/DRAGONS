
NIRI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI.py

.. code-block:: python
    :linenos:

    
    class NIRI(DataClassification):
        name="NIRI"
        usage = "Applies to any data from the NIRI instrument."
        parent = "GEMINI"
    
        requirement = PHU(INSTRUME='NIRI')
    
    newtypes.append(NIRI())



