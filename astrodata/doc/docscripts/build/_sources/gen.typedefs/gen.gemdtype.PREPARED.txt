
PREPARED Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    PREPARED

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.PREPARED.py

.. code-block:: python
    :linenos:

    class PREPARED(DataClassification):
        
        name="PREPARED"
        usage = 'Applies to all "prepared" data.'
        parent = "UNPREPARED"
        requirement = PHU( {'{re}.*?PREPAR*?': ".*?" })
        
    newtypes.append(PREPARED())



