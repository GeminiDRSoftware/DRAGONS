
NICI_ADI_B Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_ADI_B

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_ADI_B.py

.. code-block:: python
    :linenos:

    
    class NICI_ADI_B(DataClassification):
        name="NICI_ADI_B"
        usage = "Applies to imaging datasets from the NICI instrument."
        parent = "NICI"
        # DICHROIC PHU keyword value contains the string 'Mirror'
        requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?Mirror*?" })
    
    newtypes.append(NICI_ADI_B())



