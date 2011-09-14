.. faq:

**************************
Frequently Asked Questions
**************************

**1. What can the QAP do?**

   In a nutshell, it can process GMOS Imaging data.  For complete list of
   what it can do, see Features and Capabilities Currently Available.

**2. Where does the QAP run?**

   The software can be run from any Linux or Mac desktop properly configured.
   However, for night operations use mkopipe1 or hbfpipe1. 

**3. How do I start the QAP?**

   From a ``telops`` session, set up the QAP environment with ``startqap``.
   Then for each new dataset, type ``redux <image_number>``.  For more
   information, see How to operate the Gemini Quality Assurance Pipeline.

**4. Do I have to start the QAP?**

   No central process needs to be started.  At this time, there is no automatic
   detection of the new data, so the user needs to manually launch the reduction
   of each dataset with the ``redux`` command.   However, a ``ds9`` must be launched
   and the QAP must run in a specific directory.  To set that up simply
   type ``startqa`` from a ``telops`` account.  For more
   information, see How to operate the Gemini Quality Assurance Pipeline.

**5. How do I operate the QAP?**

   Once set up, simply do ``redux <image_number>``.

**6. How to process the next dataset?**

   Once set up, simply do ``redux <image_number_of_new_dataset>``.

**7. How to process the previous dataset?**

   Once set up, simply do ``redux <image_number_of_previous_dataset>``.

**8. Is the QAP interactive?**

   Nope.  PyRAF should be used for interactivity.

