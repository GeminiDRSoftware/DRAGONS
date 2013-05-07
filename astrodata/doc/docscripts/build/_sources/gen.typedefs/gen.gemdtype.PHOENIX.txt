
PHOENIX Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PHOENIX

Source Location 
    ADCONFIG_Gemini/classifications/types/PHOENIX/gemdtype.PHOENIX.py

.. code-block:: python
    :linenos:

    
    class PHOENIX(DataClassification):
        name="PHOENIX"
        usage = ""
        requirement = PHU(INSTRUME='PHOENIX')
    
    newtypes.append(PHOENIX())



