Creating a New Descriptor
!!!!!!!!!!!!!!!!!!!!!!!!!!

The descriptor implementation are defined in the
``astrodata_Sample/ADCONFIG_Sample/descriptors`` directory tree. A descriptor
configuration requires the following elements:

1. There must be a "Calculator" object in which the descriptor function must be
   defined (as a method).

2. The Calculator class must appear in a "calculator index", which are any files
   in the ``descriptors`` directory tree, named ``calculatorIndex.<whatever>.py``
   where ``<whatever>`` can be any unique name.

3. The descriptor must be listed in the ``DescriptorsList.py`` file, used to
   attach the function to appropriate AstroData instances.

The Calculator Class
@@@@@@@@@@@@@@@@@@@@@

The Calculator Class in the Sample package is in the file
``astrodata_Sample/ADCONFIG_Sample/descriptors/OBSERVED_Descriptors.py``.
It contains just one example descriptor functions, ``observatory`` which relies
on the standards MEF PHU key, ``OBSERVAT``. Full source::

    class OBSERVED_DescriptorCalc:
        def observatory(self, dataset, **args):
            return dataset.get_phu_key_value("OBSERVAT")

The Calculator Index
@@@@@@@@@@@@@@@@@@@@@

The Calculator Index for astrodata_Sample is located in the file,
``astrodata_Sample/ADCONFIG_Sample/descriptors/calculatorIndex.Sample.py``.
Here is the source::

    calculatorIndex = {
        "OBSERVED":"OBSERVED_Descriptors.OBSERVED_DescriptorCalc()",
        }
    
Note, the sample index also contains detailed instructions about the format but
for our purposes the index should be clear enough.  The dictionary key is the
string name of a defined AstroData Type, and the value is the class name,
including the module it is defined in.  The system will parse this name and
import the ``OBSERVED_Descriptors`` module, then store the class in a calculator
dictionary.

The DescriptorList.py
@@@@@@@@@@@@@@@@@@@@@@

The ``DescriptorList.py`` file contains  a list of descriptors.  The entries
declared by declaring "DD" objects that gives at the least the name of the
descriptor, and in more advanced cases also other descriptor related meta-data.

Adding a New Descriptor to the 
