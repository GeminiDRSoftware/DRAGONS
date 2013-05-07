
TRECS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    TRECS

Source Location 
    ADCONFIG_Gemini/classifications/types/TRECS/gemdtype.TRECS.py

.. code-block:: python
    :linenos:

    class TRECS(DataClassification):
        name = "TRECS"
        usage = "Applies to all datasets from the TRECS instrument"
        parent = "GEMINI"
        requirement = PHU(INSTRUME="TReCS")
    
    newtypes.append(TRECS())



