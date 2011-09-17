
LS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    LS

Source Location 
    ADCONFIG_Gemini/classifications/types/generic/gemdtype.LS.py

.. code-block:: python
    :linenos:

    class LS(DataClassification):
        name="LS"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all long slit spectral datasets.
            '''
        parent = "SPECT"
        requirement= ISCLASS("GMOS_MOS")
    
    newtypes.append(LS())



