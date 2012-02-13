import sys
import os
import re

#from nose.tools import *
from nose.plugins.skip import Skip, SkipTest

from astrodata import AstroData
from astrodata import Errors
from file_urls import sci123

# use debug to print out extra detail for failures
# '_' is appended to verbose to prevent nose collision
debug = False

# helper function
def result_handler(msg, cmsg, exval, cexval, debug, outstr):
    if msg == cmsg: 
        outstr += "Passed Test: ExpectedException\n"
        if debug:
            outstr += " " * 50 +"Except:" + msg

    elif exval is None and cexval is None:
        outstr += "Passed Test: ExceptionMsgsDiff\n"
        if debug:
            outstr += " " * 50 + "DV Test Except:" + msg
            outstr += "\n" + " " * 50 + "Control Except:" + cmsg
    elif exval is not None and cexval is None:
        outstr += "FAILED Test: Exception in Control operator, not in python"
        outstr += "\n" + " " * 50 + "DV Test Result:" + exval
        outstr += "\n" + " " * 50 + "Control Except:" + cmsg
    elif exval is None and cexval is not None:
        outstr += "FAILED Test: Exception in DV operator, not in python\n"
        outstr += " " * 50 + "Control Test Result:" + cexval
        outstr += "\n" + " "*50 + "DV Test Except:" + msg
    else:
        outstr += "FAILED Test:\n"
        outstr += " " * 50 + "DV Test Except:" + msg
        outstr += "\n" + " " * 50 + "DV Test Result:" + exval
        outstr += "\n" + " " * 50 + "Control Except:" + cmsg
        outstr += "\n" + " " * 50 + "Control Test Result:" +cexval
    return outstr



def test1():
    """ASTRODATA-descriptor-value TEST 1: Python vs. DV"""
    tstr = """Need to visit 6 Failures and no datetime testing:
     
     DESCRIPTOR: observation_id, GN-2009A-Q-2-23,  PYTYPE: <type 'str'>
    
     (1, 2, 3) dval % (stringdv, floatdv, intdv) ==> Python=GN-2009A-Q-2-23 DV=None
    
     (4, 5, 6) "'hi'" % dval ==> Python=Control Except, DV='hi' (same for float, int)
    """
    print tstr
    ad = AstroData(sci123)
    outstr = ""
    outstr += "\n\tTest input file: " + sci123
    outstr += '''\n\n** Summary: 
        A descriptor value (DV) is first tested as itself, a 
        Descriptor Value class instance, versus a data type of 
        either float, int, string, or other DVs using an overloaded 
        operator. In the table below one can see the expression
        and result to the left of "||".  The DV is then tested as
        its own pytype (dv.as_pytype) against the same operator 
        ,which is not overloaded now because it is acting as its
        python data type. The results of both tests are compared
        and reported to the right of "||" in the table below.
        The test is then repeated for every combination of dv type
        and operator. 
        \n** Interpreting Results:
        "Passed Test"
            the overloaded DV operator and python operator did not throw an
            exception and are equal.
        "Passed Test: ExpectedException" 
            both operators threw an exception with the same message 
        "Passed Test: ExceptionMsgsDiff" 
            both operators threw an exception with different messages
        "FAILED Test:..."
            both operators have completely different results.
        '''
    intdv = ad.detector_x_bin()
    floatdv = ad.exposure_time()
    stringdv = ad.raw_cc()
    outstr += "\n** DV operands:\n"
    outstr += "         intdv:" + str(intdv) + " (detector_x_bin)\n"
    outstr += "       floatdv:" + str(floatdv) + " (exposure_time)\n"
    outstr += "      stringdv:" + str(stringdv) + " (raw_cc)\n"
    outstr += "_"*80 + "\n"

    descripts = ["airmass", "amp_read_area", "azimuth", "camera",
     "cass_rotator_pa", "central_wavelength", "coadds", "data_label",
     "data_section", "dec", "decker", "detector_section", "detector_x_bin",
     "detector_y_bin", "disperser", "dispersion", "dispersion_axis",
     "elevation", "exposure_time", "filter_name", "focal_plane_mask", "gain",
     "gain_setting", "grating", "instrument", "local_time", "mdf_row_id",
     "nod_count", "nod_pixels", "non_linear_level", "object",
     "observation_class", "observation_epoch", "observation_id",
     "observation_type", "pixel_scale", "prism", "program_id", "pupil_mask",
     "qa_state", "ra", "raw_bg", "raw_cc", "raw_iq", "raw_wv", "read_mode",
     "read_noise", "read_speed_setting", "saturation_level", "slit",
     "telescope", "ut_date", "ut_datetime", "ut_time", "wavefront_sensor",
     "wavelength_reference_pixel", "well_depth_setting", "x_offset",
     "y_offset"]


    descripts = ["detector_y_bin","pixel_scale","observation_id"]      
    ops = ["+","-","*","/","//","%","**", "<<",">>", "^", "<", "<=", ">", \
            ">=","==",]
    #ops = ["%"]#,"+","**"]
    exprs = []
    operands = [10, 10., "'hi'"]
    dvoperands = ["intdv","floatdv","stringdv"]

    #Create Permutations for operators and operands
    for op in ops:
        if len(operands) > 0:
            for operand in operands:
                expr = "%(lop)s %(op)s %(rop)s" % { "lop": "dval",
                                                    "rop": repr(operand),
                                                    "op":  op}
                exprs.append(expr)
                expr = "%(lop)s %(op)s %(rop)s" % { "lop": repr(operand),
                                                    "rop": "dval",
                                                    "op":  op}
                exprs.append(expr)
        if len(dvoperands) > 0:
            for dvo in dvoperands:
                expr = "%(lop)s %(op)s %(rop)s" % { "lop": "dval",
                                                    "rop": dvo,
                                                    "op":  op}
                exprs.append(expr)
                expr = "%(lop)s %(op)s %(rop)s" % { "lop": dvo,
                                                    "rop": "dval",
                                                    "op":  op}
                exprs.append(expr)
        expr = "%(lop)s %(op)s %(rop)s" % { "lop": "dval",
                                            "rop": "dval",
                                            "op":  op}
        exprs.append(expr)

    # Test permutations of expr with descriptor values
    for desc in descripts:
        outstr += "\n     DESCRIPTOR: %s" % desc
        try:
            dval = eval("ad.%s()"%desc)
            outstr += "\n" + "DESCRIPTOR VALUE: " + str(dval.as_pytype())
            pydval = dval.pytype(dval)
            outstr += "\n" + "          PYTYPE: " +  repr(dval.pytype) + "\n\n"
            for expr in exprs:
                msg=" "
                cmsg=" "
                try:
                    le = len(expr)
                    exval = eval(expr)
                    exvalAsString = str(exval)
                    if len(exvalAsString) > 15:
                        exvalAsString = exvalAsString[0:11] + "..."
                    oute = "%s%s%s%s" % ( expr,
                                            " "*(20 - le),
                                            " ==> ", 
                                            exvalAsString)
                    
                    oute += " " * (45 - len(oute))
                except Errors.IncompatibleOperand:
                    oute = "IncompatibleOperand: type(dval)= %s " % \
                        str(dval.pytype) + expr
                    exval = "FAILED!"
                    outstr += oute
                except TypeError, e:
                    oute = "(%s) TypeError: %s" % (expr,str(e))
                    exval = "FAILED!"
                    msg = str(e)
                except Errors.DescriptorValueTypeError, e:
                    le = len(expr)
                    oute = "%s%s%s" % ( expr,
                                          " " * (20 - le),
                                          " ==> DVtypeError")
                    exval = None
                    msg = str(e)
                    oute += " " * (45 - len(oute))
                except OverflowError, e:
                    le = len(expr)
                    oute = '%s%s%s' % ( expr,
                                          ' ' * (20 - le),
                                          ' ==> OverflowError')
                    exval = None
                    msg = str(e)
                    oute += ' ' * (45 - len(oute))
                    
                outstr += oute + "||"
                try:
                    controlexpr = re.sub("dval", repr(pydval), expr)
                    cexval = eval(controlexpr)
                except TypeError, e:
                    cmsg = str(e)
                    cexval = None
                    outstr = result_handler(msg, cmsg, exval, cexval , \
                        debug, outstr)
                    continue
                except Errors.DescriptorValueTypeError, e:
                    csmg = str(e)
                    cexval = None
                    outstr = result_handler(msg, cmsg, exval, cexval , \
                        debug, outstr)
                    continue
                except Errors.IncompatibleOperand, e:
                    csmg = str(e)
                    cexval = None
                    outstr = result_handler(msg, cmsg, exval, cexval , \
                        debug, outstr)
                    continue
                except:
                    cexval = None
                    outstr = result_handler(msg, cmsg, exval, cexval , \
                        debug, outstr)
                    continue
                
                if type(cexval) != type(exval):
                    outstr += "FAILED Test: Types Differ " + \
                        str(type(exval)) + str(type(cexval))
                    outstr += "\n" + " " * 50 + "   Test Result = " + \
                        repr(exval) 
                    outstr += "\n" + " " * 50 + "Control Result = " + \
                        repr(cexval) + "\n"
                elif cexval != exval:
                    outstr += "FAILED Test: Results Differ "  + str(exval) + \
                        " != " + str(cexval)
                    outstr += " " * 50, "   Test Result = " + exval 
                    outstr += " " * 50, "Control Result = " + cexval
                else:
                    outstr += "Passed Test\n"

            exec("%s = dval" % desc)
        except KeyError:
            outstr += "FAILED due to KeyError!" #, ad.exception_info
        except Errors.DescriptorTypeError:
            outstr += "FAILED with DescriptorTypeError"
        except Errors.ExistError:
            outstr += "FAILED with ExistError"
        except:
            outstr += "FAILED"
            raise
        outstr += "_"*80 + "\n"
    print outstr        
    
    # Use re module to calculate report
    output_string = outstr
    passtotal = re.compile("\|\|Passed Test")
    pass_total = len(passtotal.findall(output_string))
    passwithemd = re.compile("\|\|Passed Test: ExceptionMsgsDiff")
    pass_with_emd = len(passwithemd.findall(output_string))
    passwithee = re.compile("\|\|Passed Test: ExpectedException")
    pass_with_ee = len(passwithee.findall(output_string))
    fail = re.compile("\|\|FAILED Test")
    fail_ = len(fail.findall(output_string))
    pass_without_e = pass_total - (pass_with_emd + pass_with_ee)
    
    # print REPORT
    print("Ran %i operator comparison tests" % (pass_total + fail_))
    if pass_with_ee > 0:
        print("\t(%i passed with the same expected exceptions)" % \
            pass_with_ee)
    if pass_with_emd > 0:
        print("\t(%i passed with different expected exceptions)" % \
            pass_with_emd)
    if fail_ > 0:
        print("%i FAILED"  % fail_)
        print("\t(run in debug mode to see more details)")
    print("_"*80 + "\n\n")

