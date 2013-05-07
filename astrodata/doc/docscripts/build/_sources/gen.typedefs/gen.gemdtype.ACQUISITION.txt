
ACQUISITION Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    ACQUISITION

Source Location 
    ADCONFIG_Gemini/classifications/types/gemdtype.ACQUISITION.py

.. code-block:: python
    :linenos:

    class ACQUISITION(DataClassification):
        name="ACQUISITION"
        usage = """
            Applies to all Gemini acquisitions
            """
        parent = "GENERIC"
        requirement = OR([  PHU(OBSCLASS="acq"),
                            PHU(OBSCLASS="acqCal")  ])
    
    newtypes.append(ACQUISITION())



