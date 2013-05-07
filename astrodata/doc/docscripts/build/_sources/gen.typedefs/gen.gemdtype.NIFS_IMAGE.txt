
NIFS_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIFS_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/NIFS/gemdtype.NIFS_IMAGE.py

.. code-block:: python
    :linenos:

    
    class NIFS_IMAGE(DataClassification):
        name="NIFS_IMAGE"
        usage = "Applies to any image dataset from the NIFS instrument."
        parent = "NIFS"
    
        requirement = ISCLASS("NIFS") & PHU( FLIP='In')
    
    newtypes.append(NIFS_IMAGE())



