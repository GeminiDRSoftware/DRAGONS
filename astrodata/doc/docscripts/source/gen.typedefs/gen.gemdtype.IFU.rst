
IFU Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    IFU

Source Location 
    ADCONFIG_Gemini/classifications/types/generic/gemdtype.IFU.py

.. code-block:: python
    :linenos:

    class IFU(DataClassification):
        name="IFU"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all Gemini IFU data.
            '''
        parent = "SPECT"
        requirement = ISCLASS("GMOS_IFU")
    
    newtypes.append( IFU())



