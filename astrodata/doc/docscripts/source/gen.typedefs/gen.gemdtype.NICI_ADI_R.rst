
NICI_ADI_R Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_ADI_R

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_ADI_R.py

.. code-block:: python
    :linenos:

    
    class NICI_ADI_R(DataClassification):
        name="NICI_ADI_R"
        usage = "Applies to imaging datasets from the NICI instrument."
        parent = "NICI"
        # DICHROIC PHU keyword value contains the string 'Open'
        requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?Open*?" })
    
    newtypes.append(NICI_ADI_R())



