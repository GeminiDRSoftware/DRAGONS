
NICI_IMAGE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_IMAGE

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_IMAGE.py

.. code-block:: python
    :linenos:

    
    class NICI_IMAGE(DataClassification):
        name="NICI_IMAGE"
        usage = "Applies to imaging datasts from the NICI instrument."
        parent = "NICI"
        requirement = ISCLASS('NICI') & PHU(INSTRUME='NICI')
    
    newtypes.append(NICI_IMAGE())



