
UNPREPARED Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    UNPREPARED

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.UNPREPARED.py

.. code-block:: python
    :linenos:

    class UNPREPARED(DataClassification):
        editprotect=True
        name="UNPREPARED"
        usage = 'Applies to un-"prepared" datasets, datasets which have not had the prepare task run on them.'
        parent = "RAW"
        requirement= PHU({'{prohibit,re}.*?PREPAR*?': ".*?" })
        
    newtypes.append(UNPREPARED())



