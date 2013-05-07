
SIDEREAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    SIDEREAL

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.SIDEREAL.py

.. code-block:: python
    :linenos:

    
    class SIDEREAL(DataClassification):
        name="SIDEREAL"
        usage = "Data taken with the telesocope tracking siderealy"
        
        parent = "GEMINI"
        requirement = PHU(DECTRACK='0.') & PHU(RATRACK='0.')
    
    newtypes.append(SIDEREAL())



