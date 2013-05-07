Creating a New Descriptor
!!!!!!!!!!!!!!!!!!!!!!!!!

The Descriptor implementations are defined in the
``astrodata_Sample/ADCONFIG_Sample/descriptors`` directory tree. A descriptor
configuration requires the following elements:

1. There must be a "Calculator" object in which the descriptor function must be
   defined (as a method).

2. The Calculator class must appear in a "calculator index", which are any files
   in the ``descriptors`` directory tree named ``calculatorIndex.<whatever>.py``
   where ``<whatever>`` can be any unique name.

3. The descriptor must be listed in the ``DescriptorsList.py`` file.

The Calculator Class
@@@@@@@@@@@@@@@@@@@@

The ``Calculator`` Class in the Sample package is, for the OBSERVED type, in the file
``OBSERVED_Descriptors.py``. That file is located in the ``descriptors`` subdirectory of the
``ADCONFIG_Sample`` of the ``astrodata_Sample`` package.
It contains just one example descriptor function, ``observatory`` which relies
on the standard MEF PHU key, ``OBSERVAT``::

    class OBSERVED_DescriptorCalc:
        def observatory(self, dataset, **args):
            return dataset.get_phu_key_value("OBSERVAT")
            
In order for this function to be called for the right type of data, this class
must appear in a "calculator index".

The Calculator Index
@@@@@@@@@@@@@@@@@@@@

The Calculator Index for astrodata_Sample is located in the file,
``calculatorIndex.Sample.py``, in the ``descriptors`` subdirectory of 
``ADCONFIG_Sample`` in the ``astrodata_Sample`` configuration package.

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
@@@@@@@@@@@@@@@@@@@@@

The ``DescriptorList.py`` file contains  a list of descriptors definitions.
The entries
declared must at least declare the name of the new descriptor function.  The
infrastructure will use these names to create a bridge between AstroData instances and
the type-specific descriptor functions.

Adding a New Descriptor to the configuration involves:

#. Adding a "DescriptorDescriptor" to the DescriptorList.py file.
#. Adding the descriptor function to the appropriate Descriptor Calculator class.

The DescriptorList.py File
##########################

The contents of ``DescriptorList.py`` is a list of "DD" object constructors, as follows
from the astrodata_Sample package::

    [
      DD("observatory"),
    ]

To add a descriptor named "telescope" we'd add the following line to the ``DescriptorList.py`` file::

    DD("telescope")
    
This tells the infrastructure the name of the descriptor, and in more complicated cases can provide other descriptor
metadata to the infrastructure.  The final file would look as follows::

    [
      DD("observatory"),
      DD("telescope")
     ]
     
Adding the Descriptor Function to the CalculatorClass
#####################################################

To add the descriptor once the descriptor is present in the ``DescriptorList.py`` one merely needs to add a function to the
appropriate DescriptorCalculator class. The contents of
``OBSERVED_Descriptors.py`` module in the astrodata_Sample
configuration is::

    class OBSERVED_DescriptorCalc:
        def observatory(self, dataset, **args):
            return dataset.get_phu_key_value("OBSERVAT")
        
To add the "telescope" descriptor means adding another function to this class::

        def telescope(self, dataset, **args):
            return dataset.get_phu_key_value("TELESCOP")

All descriptors should have the same function signature,  including
``self``, a ``dataset`` argument and ``**args`` to catch all named arguments.
The latter is required by the infrastructure so that unexpected parameters 
can be sent to all descriptor algorithms, some of which may be handled by the
infrastructure on behalf of the descriptor function.
