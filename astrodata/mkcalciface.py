"""
Module to find, create and or write CalculatorInterface modules / classes for
package descriptor lists.

"""
import imp
import os
import re

from datetime import datetime
from astrodata import Errors


CALCULATOR_INTERFACE_MARKER = "CalculatorInterface"
CALCULATOR_FILE_MARKER = "CALCIFACE"
DESCRIPTOR_LIST_FILE_MARKER = "DDLIST"


#The spacing in the FUNCTION_BUFFER must be retained
FUNCTION_BUFFER = """
    def %(name)s(self, format=None, **args):
        \"\"\"%(description)s\"\"\"
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = \"key_%(name)s\"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, \"%(name)s\"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, \"exception_info\"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = (\"Unable to find an appropriate descriptor \"
                           \"function or a default keyword for '%(name)s'\")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.%(name)s(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            %(pytypeimport)s
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = \"%(name)s\",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = %(pytype)s )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, \"exception_info\"):
                    setattr(self, \"exception_info\", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = \"%(name)s\",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise
"""

# Spacing in WHOLE_OUT must be retained
WHOLE_OUT = (
"""import sys
from astrodata import Descriptors
from astrodata.Descriptors import DescriptorValue
from astrodata import Errors

class CalculatorInterface(object):

    descriptor_calculator = None
%(descriptors)s
    # UTILITY FUNCTIONS, above are descriptor function buffers
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptors.get_calculator(self, **args)
""")


def get_calculator_interface():
    """Combination of making and getting calculator interafce objects"""
    from astrodata.ConfigSpace import ConfigSpace

    # Loop over found interface (type, file) markers found in the name space
    calculator_interfaces = []
    pkgs_calculator_interfaces = ConfigSpace.calc_iface_list
    for interface_type, interface_file in pkgs_calculator_interfaces:

        # Already created CalculatorInterface from an astrodata package
        if interface_type == CALCULATOR_FILE_MARKER:
            obj = load_class_from_file(interface_file,
                                       CALCULATOR_INTERFACE_MARKER)

            if obj is not None:
                calculator_interfaces.append(obj.__class__)

        elif interface_type == DESCRIPTOR_LIST_FILE_MARKER:
            # Set path for doc strings
            interface_path = os.path.dirname(interface_file)
            if interface_path == '':
                interface_path = None

            # Descriptor list file - Create a new calculator interface
            cib = open(interface_file)
            cibsrc = cib.read()
            cib.close()
            descriptor_dict = {"DescriptorDescriptor": DD,
                               "DD": DD,
                               "datetime": datetime
                               }
            ddlist = eval(cibsrc, descriptor_dict)

            try:
                cisrc = mk_calc_iface_body(ddlist, interface_path)
            except Errors.BadConfiguration as bc_err:
                bc_err.add_msg("FATAL CONFIG ERROR: %s" % interface_file)
                raise bc_err

            exec(cisrc, descriptor_dict)
            for key in descriptor_dict:
                if re.match(CALCULATOR_INTERFACE_MARKER, key):
                    calculator_interfaces.append(descriptor_dict[key])

    # Add ComplexCalculatorInterface as base class
    calculator_interfaces.append(ComplexCalculatorInterface)

    # Return new caclulator interface class using new style classes
    return type('CalculatorInterface', tuple(calculator_interfaces), {})


def mk_calc_iface_body(ddlist, path=None):
    """
    Create descriptor functions that form the bulk of the CalculatorInterface
    body

    """
    d_descriptor = None
    try:
        out = ''.join([d_descriptor.funcbody(path) for d_descriptor in ddlist])
    except Errors.BadConfiguration as bc_err:
        if d_descriptor is not None:
            bc_err.add_msg("Problem with ddlist item #%d" %
                           ddlist.index(d_descriptor))
        raise bc_err

    finalout = WHOLE_OUT % {"descriptors": out}

    return finalout


class ComplexCalculatorInterface(object):
    """
    Empty class to allow basing on new style classes and allowing for future
    extention

    """
    pass


class DescriptorDescriptor(object):
    """
    Individual descriptor class to create the function string buffer for a
    descriptor to be used in the CalculatorInterface

    """
    def __init__(self, name=None, pytype=None, path=None):
        self.name = name
        self.pytype = pytype
        self.path = self._set_path(path)

    def _set_path(self, path):
        """
        Set path to module containing descriptor list or calculator
        interface

        """
        if path is None:
            self.path = "."
        else:
            self.path = path

    def funcbody(self, path):
        """ Create the body of the descriptor function """
        self._set_path(path)

        if self.pytype:
            pytypestr = self.pytype.__name__
        else:
            pytypestr = "None"

        if pytypestr == "datetime":
            pti = "from datetime import datetime"
        else:
            pti = ""

        if self.pytype is not None:
            rtype = self.pytype.__name__
            if rtype == 'str':
                rtype = 'string'
            if rtype == 'int':
                rtype = 'integer'
        else:
            raise Errors.BadConfiguration("'DD' Declaration for '%s' "
                                          "needs pytype defined." % self.name)

        # Use the docstring defined in the docstrings module, if it exists
        # TODO Import this only once per descriptor list
        # Make strings in this call globals / constansts
        docstrings = load_class_from_file(os.path.join(self.path,
                                                       'docstrings.py'),
                                          'docstrings')
        if docstrings is not None and hasattr(docstrings, self.name):
            doc = getattr(docstrings, self.name)
            description = doc.__doc__
        else:
            doc = ("\n"
                   "        Return the %(name)s value\n\n"
                   "        :param dataset: the data set\n"
                   "        :type dataset: AstroData\n"
                   "        :param format: the return format\n"
                   "        :type format: string\n"
                   "        :rtype: %(rtype)s as default (i.e., format=None)\n"
                   "        :return: the %(name)s value\n"
                   "        ") % {'name': self.name, 'rtype': rtype}
            description = doc

        ret = FUNCTION_BUFFER % {'name': self.name,
                                 'pytypeimport': pti,
                                 'pytype': pytypestr,
                                 'description': description}
        return ret


DD = DescriptorDescriptor


def load_class_from_file(filepath, requested_class):
    """
    Dynamically load a class from a file on disk (either source or byte
    compiled)

    Parameters
    ----------

    `filepath`: filename including path to file

    `requested_class`: Name of class to be imported

    Returns
    -------

    Imported class if found or None if not present

    ##TODO: Wrap the imports in try except clauses
    ##TODO: Move to a more centralised 'utilities' module
    """
    class_instance = None
    module_name, file_ext = os.path.splitext(os.path.split(filepath)[-1])

    _module = None
    if file_ext.lower() == '.py':
        _module = imp.load_source(module_name, filepath)

    elif file_ext.lower() == '.pyc':
        _module = imp.load_compiled(module_name, filepath)

    if _module is not None and hasattr(_module, requested_class):
        class_instance = getattr(_module, requested_class)()

    return class_instance
