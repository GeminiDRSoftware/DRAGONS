
GMOS_RAW Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS_RAW

Source Location 
    ADCONFIG_Gemini/classifications/status/gemdtype.GMOS_RAW.py

.. code-block:: python
    :linenos:

    class GMOS_RAW(DataClassification):
        editprotect=True
        name="GMOS_RAW"
        usage = 'Applies to RAW GMOS data.'
        typeReqs= ['GMOS']
        requirement = ISCLASS("RAW") & ISCLASS("GMOS")
        
    newtypes.append(GMOS_RAW())



