
NIRI_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI_IMAGE.py

.. code-block:: python
    :linenos:

    
    class NIRI_IMAGE(DataClassification):
        name="NIRI_IMAGE"
        usage = "Applies to any IMAGE dataset from the NIRI instrument."
        parent = "NIRI"
        requirement = ISCLASS('NIRI') & PHU({"{prohibit}FILTER3":"(.*?)grism(.*?)"})
    
    
    newtypes.append(NIRI_IMAGE())



