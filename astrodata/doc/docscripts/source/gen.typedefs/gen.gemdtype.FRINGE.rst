
FRINGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    FRINGE

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.FRINGE.py

.. code-block:: python
    :linenos:

    class FRINGE(DataClassification):
        name="FRINGE"
        usage = "A processed fringe."
        parent = "CAL"
        requirement = PHU(GIFRINGE='(.*?)')
    
    newtypes.append(FRINGE())



