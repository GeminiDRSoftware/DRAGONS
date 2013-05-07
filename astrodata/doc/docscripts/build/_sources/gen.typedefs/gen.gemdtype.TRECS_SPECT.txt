
TRECS_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    TRECS_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/TRECS/gemdtype.TRECS_SPECT.py

.. code-block:: python
    :linenos:

    class TRECS_SPECT(DataClassification):
        name = "TRECS_SPECT"
        usage = "Applies to all spectroscopic datasets from the TRECS instrument"
        parent = "TRECS"
        requirement = ISCLASS("TRECS") & NOT(ISCLASS("TRECS_IMAGE"))
    
    newtypes.append(TRECS_SPECT())



