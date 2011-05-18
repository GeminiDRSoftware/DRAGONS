import sys, os
from astrodata import AstroData
from astrodata import Errors
import re
from optparse import OptionParser

parser = OptionParser()
parser.set_description(
"""Descriptor Value Operators Test.\n Created by C.Allen, 
K.Dement, Apr2011.""" )

parser.add_option('-v', '--verbose', action='store_true', dest='verbose', 
                    default=False, help='verbose testing')
parser.add_option('-w', '--writefile', action='store_true', dest='writefile',
                    default=False, help='verbose testing')
(options,  args) = parser.parse_args()

verbose = options.verbose

def resultHandler(msg,cmsg,exval,cexval,verbose,outstr):
    if msg == cmsg: 
        outstr += "Passed Test: ExpectedException\n"
        if verbose:
            outstr += " "*50 +"Except:" + msg

    elif exval is None and cexval is None:
        outstr += "Passed Test: ExceptionMsgsDiff\n"
        if verbose:
            outstr += " "*50 + "DV Test Except:" + msg
            outstr += "\n" + " "*50 + "Control Except:" + cmsg
    elif exval is not None and cexval is None:
        outstr += "FAILED Test: Passed DV Test but threw Exception in Control\n"
        outstr += " "*50 + "DV Test Result:" + exval
        outstr += "\n" + " "*50 + "Control Except:" + cmsg
    elif exval is None and cexval is not None:
        outstr += "FAILED Test: Passed Control Test but threw Exception in DV Test\n"
        outstr += " "*50 + "Control Test Result:" + cexval
        outstr += "\n" + " "*50 + "DV Test Except:" + msg
    else:
        outstr += "FAILED Test:\n"
        outstr += " "*50 + "DV Test Except:" + msg
        outstr += "\n" + " "*50 + "DV Test Result:" + exval
        outstr += "\n" + " "*50 + "Control Except:" + cmsg
        outstr += "\n" + " "*50 + "Control Test Result:" +cexval
    return outstr

# overloaded operators test
outstr = ""
testdatafile = ""
outstr += "_"*80 + "\n\n\n\t\tADTEST: DV OVERLOADED OPERATORS TEST\n\n\n\n"
if len(args) > 1:
    "Too many arguments!"
    raise

if len(args) is 1:
    ad = AstroData(args[0])
    outstr += "** ad Testdata: "+args[0]
else:
    testdatafile = "../../../../test_data/recipedata/N20090703S0163.fits"
    ad = AstroData(testdatafile)
    outstr += "** Testdata: "+testdatafile
# the rtf will use the following file:
#ad = AstroData('/rtfperm/rtftests/gemini_iraf/gmosspec2/orig/S20100223S0046.fits')

outstr += '''\n\n** Description: A DV is tested as an instance first.
   The evaluated expression and ==> result appear to
   the left of "||".  The DV is then tested as its own
   pytype, the results of which are compared against
   the instance test result and reported after the '||'\n 
** Interpreting Results:
   "Passed Test" means the comparison values are the same. 
   "Passed Test: ExpectedException" means both tests 
       threw an exception with the same message 
   "Passed Test: ExceptionMsgsDiff" means both tests 
       threw an exception with different messages
   "Failed Test:..." means something is wrong with the 
       overloaded operator\n'''
intdv = ad.detector_x_bin()
floatdv = ad.exposure_time()
stringdv = ad.raw_cc()
outstr += "** DV operands:\n"
outstr += "         intdv:" + str(intdv) + " (detector_x_bin)\n"
outstr += "       floatdv:" + str(floatdv) + " (exposure_time)\n"
outstr += "      stringdv:" + str(stringdv) + " (raw_cc)\n"
outstr += "_"*80 + "\n"

descripts = ["airmass", "amp_read_area", "azimuth", "camera", "cass_rotator_pa",
 "central_wavelength", "coadds", "data_label", "data_section", "dec", "decker",
 "detector_section", "detector_x_bin", "detector_y_bin", "disperser",
 "dispersion", "dispersion_axis", "elevation", "exposure_time", "filter_name",
 "focal_plane_mask", "gain", "gain_setting", "grating", "instrument",
 "local_time", "mdf_row_id", "nod_count", "nod_pixels", "non_linear_level",
 "object", "observation_class", "observation_epoch", "observation_id",
 "observation_type", "pixel_scale", "prism", "program_id", "pupil_mask",
 "qa_state", "ra", "raw_bg", "raw_cc", "raw_iq", "raw_wv", "read_mode",
 "read_noise", "read_speed_setting", "saturation_level", "slit", "telescope",
 "ut_date", "ut_datetime", "ut_time", "wavefront_sensor",
 "wavelength_reference_pixel", "well_depth_setting", "x_offset", "y_offset"]

descripts = ["detector_y_bin","pixel_scale","observation_id"]      
ops = ["+","-","*","/","//","%","**", "<<",">>", "^", "<", "<=", ">",">=","==",]
#ops = ["+","%","**"]
exprs = []
operands = ['"hi"', 10., 10.]
dvoperands = ["intdv","floatdv","stringdv"]

#Create Permutations for operators and operands
for op in ops:
    for operand in operands:
        expr = "%(lop)s %(op)s %(rop)s" % { "lop": "dval",
                                            "rop": repr(operand),
                                            "op":  op}
        exprs.append(expr)
        expr = "%(lop)s %(op)s %(rop)s" % { "lop": repr(operand),
                                            "rop": "dval",
                                            "op":  op}
        exprs.append(expr)
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
                                        " "*(20-le),
                                        " ==> ", 
                                        exvalAsString)
                
                oute += " "*(45-len(oute))
            except Errors.IncompatibleOperand:
                oute = "IncompatibleOperand: type(dval)= %s " % str(dval.pytype) + expr
                exval = "FAILED!"
                outstr += oute
            except TypeError, e:
                oute = "(%s) TypeError: %s" %(expr,str(e))
                exval = "FAILED!"
                msg = str(e)
            except Errors.DescriptorValueTypeError, e:
                le = len(expr)
                oute = "%s%s%s" % ( expr,
                                      " "*(20-le),
                                      " ==> DVtypeError")
                exval = None
                msg = str(e)
                oute += " "*(45-len(oute))
            except OverflowError, e:
                le = len(expr)
                oute = '%s%s%s' % ( expr,
                                      ' '*(20-le),
                                      ' ==> OverflowError')
                exval = None
                msg = str(e)
                oute += ' '*(45-len(oute))
                
            outstr += oute +"||"
            try:
                controlexpr = re.sub("dval", repr(pydval), expr)
                cexval = eval(controlexpr)
            except TypeError, e:
                cmsg = str(e)
                cexval = None
                outstr = resultHandler(msg,cmsg,exval,cexval,verbose,outstr)
                continue
            except Errors.DescriptorValueTypeError, e:
                csmg = str(e)
                cexval = None
                outstr = resultHandler(msg,cmsg,exval,cexval,verbose,outstr)
                continue
            except Errors.IncompatibleOperand, e:
                csmg = str(e)
                cexval = None
                outstr = resultHandler(msg,cmsg,exval,cexval,verbose,outstr)
                continue
            except:
                cexval = None
                outstr += resultHandler(msg,cmsg,exval,cexval,verbose,outstr)
                continue
            
            if type(cexval) != type(exval):
                outstr += "FAILED Test: Types Differ " + str(type(exval)) + str(type(cexval))
                outstr += "\n" + " "*50 + "   Test Result = " + repr(exval) 
                outstr += "\n" + " "*50 + "Control Result = " + repr(cexval) + "\n"
            elif cexval != exval:
                outstr += "FAILED Test: Results Differ "  + str(exval) + " != "+str(cexval)
                outstr += " "*50,"   Test Result = " + exval 
                outstr += " "*50,"Control Result = " + cexval
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
    if options.writefile:
        outfile = open("output_test_dv_opterators.txt","w")
        outfile.write(outstr)
        outfile.close()
    else:
        print outstr
    



