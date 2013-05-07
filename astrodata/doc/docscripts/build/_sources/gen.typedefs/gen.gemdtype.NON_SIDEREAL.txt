
NON_SIDEREAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NON_SIDEREAL

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.NON_SIDEREAL.py

.. code-block:: python
    :linenos:

    
    class NON_SIDEREAL(DataClassification):
        name="NON_SIDEREAL"
        usage = "Data taken with the telesocope not tracking siderealy"
        
        parent = "GEMINI"
        requirement =  NOT(ISCLASS("SIDEREAL"))
    
    newtypes.append(NON_SIDEREAL())



