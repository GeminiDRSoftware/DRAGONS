
GNIRS_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GNIRS_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/GNIRS/gemdtype.GNIRS_IMAGE.py

.. code-block:: python
    :linenos:

    
    class GNIRS_IMAGE(DataClassification):
        name="GNIRS_IMAGE"
        usage = "Applies to any IMAGE dataset from the GNIRS instrument."
        parent = "GNIRS"
        requirement = ISCLASS('GNIRS') & PHU(ACQMIR='In')
    
    newtypes.append(GNIRS_IMAGE())



