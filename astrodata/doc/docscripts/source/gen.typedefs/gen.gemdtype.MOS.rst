
MOS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    MOS

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.MOS.py

.. code-block:: python
    :linenos:

    class MOS(DataClassification):
        name="MOS"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = """
            Applies to all MOS data which conformed to the required Gemini Generic
            MOS Standard
            """
        parent = "SPECT"
        requirement = OR([  ISCLASS("F2_MOS"),
                            ISCLASS("GMOS_MOS"),])
    
    newtypes.append(MOS())



