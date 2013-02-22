import sys

import numpy as np
from astrodata import AstroData
from astrodata.adutils import gemLog
from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at
from astrodata import Lookups

import pywcs

from primitives_NIRI import NIRIPrimitives

class NIRI_IMAGEPrimitives(NIRIPrimitives):
    """
    This is the class containing all of the primitives for the NIRI_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'NIRIPrimitives'.
    """
    astrotype = "NIRI_IMAGE"
    
    def init(self, rc):
        NIRIPrimitives.init(self, rc)
        return rc
    
    
    def addReferenceCatalog(self, rc):
        """
        The reference catalog is a dictionary in jhk_catalog.py


        Append the catalog as a FITS table with extenstion name
        'REFCAT', containing the following columns:

        - 'Id'       : Unique ID. Simple running number
        - 'Name'     : SDSS catalog source name
        - 'RAJ2000'  : RA as J2000 decimal degrees
        - 'DEJ2000'  : Dec as J2000 decimal degrees
        - 'J'     : SDSS u band magnitude
        - 'e_umag'   : SDSS u band magnitude error estimage
        - 'H'     : SDSS g band magnitude
        - 'e_gmag'   : SDSS g band magnitude error estimage
        - 'rmag'     : SDSS r band magnitude
        - 'e_rmag'   : SDSS r band magnitude error estimage
        - 'K'     : SDSS i band magnitude
        - 'e_imag'   : SDSS i band magnitude error estimage

        :param source: Source catalog to query. This used as the catalog
                       name on the vizier server
        :type source: string

        :param radius: The radius of the cone to query in the catalog, 
                       in degrees. Default is 4 arcmin
        :type radius: float
        """

        import pyfits as pf

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addReferenceCatalog", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addReferenceCatalog"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the necessary parameters from the RC
        source = rc["source"]
        radius = rc["radius"]

        # Get Local JHK catalog as a dictionary

        jhk = Lookups.get_lookup_table("Gemini/NIRI/jhk_catalog", "jhk") 

        #form arrays with input dict 
        ra=[]; dec=[]; vals=[]
        for key in jhk.keys():    
            ra.append(key[0])
            dec.append(key[1])
            vals.append(jhk[key])
        # sort in ra
        order = np.argsort(ra)
        ra,dec = map(np.asarray, (ra,dec))
        ra = ra[order]
        dec = dec[order]
        vals = [vals[k] for k in order]
        # Get the magnitudes and errs from each record (j,je,h,he,k,ke,name)
        vals = np.asarray([vals[k][:6] for k in range(len(ra))])
        # Separate mags into J,H,K mags arrays for clarity
        irmag={}
        irmag['Jmag']=     vals[:,0]
        irmag['Jmag_err']= vals[:,1]
        irmag['Hmag']=     vals[:,2]
        irmag['Hmag_err']= vals[:,3]
        irmag['Kmag']=     vals[:,4]
        irmag['Kmag_err']= vals[:,5]

        #print 'JMAG00:',[(irmag['Jmag'][i],irmag['Jmag_err'][i]) 
        #                for i in range(5)]

        # Loop over each input AstroData object in the input list
        adinput = rc.get_inputs_as_astrodata()
        for ad in adinput:

            try:
                input_ra = ad.ra().as_pytype()
                input_dec = ad.dec().as_pytype()
            except:
                if "qa" in rc.context:
                    log.warning("No RA/Dec in header of %s; cannot find "\
                                "reference sources" % ad.filename)
                    adoutput_list.append(ad)
                    continue
                else:
                    raise

            table_name = 'jhk.tab'
            # Loop through the science extensions
            for sciext in ad['SCI']:
                extver = sciext.extver()

                # Did we get anything?
                if (1): # We do have a dict with ra,dec
                    # Create on table per extension

                    # Create a running id number
                    refid=range(1, len(ra)+1)

                    # Make the pyfits columns and table
                    c1 = pf.Column(name="Id",format="J",array=refid)
                    c3 = pf.Column(name="RAJ2000",format="D",unit="deg",array=ra)
                    c4 = pf.Column(name="DEJ2000",format="D",unit="deg",array=dec)
                    c5 = pf.Column(name="Jmag",format="E",array=irmag['Jmag'])
                    c6 = pf.Column(name="e_Jmag",format="E",array=irmag['Jmag_err'])
                    c7 = pf.Column(name="Hmag",format="E",array=irmag['Hmag'])
                    c8 = pf.Column(name="e_Hmag",format="E",array=irmag['Hmag_err'])
                    c9 = pf.Column(name="Kmag",format="E",array=irmag['Kmag'])
                    c10= pf.Column(name="e_Kmag",format="E",array=irmag['Kmag_err'])
                    col_def = pf.ColDefs([c1,c3,c4,c5,c6,c7,c8,c9,c10])
                    tb_hdu = pf.new_table(col_def)

                    # Add comments to the REFCAT header to describe it.
                    tb_hdu.header.add_comment('Source catalog derived from the %s'
                                         ' catalog on vizier' % table_name)

                    tb_ad = AstroData(tb_hdu)
                    tb_ad.rename_ext('REFCAT', extver)

                    if(ad['REFCAT',extver]):
                        log.fullinfo("Replacing existing REFCAT in %s" % ad.filename)
                        ad.remove(('REFCAT', extver))
                    else:
                        log.fullinfo("Adding REFCAT to %s" % ad.filename)
                    ad.append(tb_ad)

            # Match the object catalog against the reference catalog
            # Update the refid and refmag columns in the object catalog
            if ad.count_exts("OBJCAT")>0:
                ad = _match_objcat_refcat(adinput=ad)[0]
            else:
                log.warning("No OBJCAT found; not matching OBJCAT to REFCAT")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

#############################################################
# Helper function for NIRI jhk catalog
def _match_objcat_refcat(adinput=None):
    """
    Match the sources in the objcats against those in the corresponding
    refcat. Update the refid column in the objcat with the Id of the 
    catalog entry in the refcat. Update the refmag column in the objcat
    with the magnitude of the corresponding source in the refcat in the
    band that the image is taken in.

    :param adinput: AD object(s) to match catalogs in
    :type adinput: AstroData objects, either a single instance or a list
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:

        # Loop over each input AstroData object in the input list
        for ad in adinput:
            filter_name = ad.filter_name(pretty=True).as_pytype()
            if filter_name in ['J', 'H', 'K']:
                magcolname = filter_name+'mag'
                magerrcolname = 'e_'+filter_name+'mag'
            else:
                log.warning("Filter %s is not in JHK table - will not be "
                            "able to flux calibrate" % filter_name)
                magcolname = None
                magerrcolname = None

                # Loop through the objcat extensions
            if ad['OBJCAT'] is None:
                raise Errors.InputError("Missing OBJCAT in %s" % (ad.filename))
            for objcat in ad['OBJCAT']:
                extver = objcat.extver()

                # Check that a refcat exists for this objcat extver
                refcat = ad['REFCAT',extver]
                if(not(refcat)):
                    log.warning("Missing [REFCAT,%d] in %s - Cannot match objcat"
                                " against missing refcat" % (extver,ad.filename))
                else:
                    # Get the x and y position lists from both catalogs
                    cat_ra = objcat.data['X_WORLD']    # this is RA
                    cat_dec = objcat.data['Y_WORLD']    # this is DEC
                    ref_ra = refcat.data['RAJ2000']
                    ref_dec = refcat.data['DEJ2000']

                    g=np.where((cat_ra<194.28) & (cat_ra>194.26))
                    #print 'RF0:',cat_ra[g]
                    #print 'RF1:',cat_dec[g]
                    # Get new pixel coordinates for all ra,dec in the dictionary.
                    # Use the input wcs object.
                   
                    # FIXME - need to address the wraparound problem here
                    # if we straddle ra = 360.00 = 0.00

                    initial = 15.0/3600.0 # 15 arcseconds in degrees
                    final = 0.5/3600.0 # 0.5 arcseconds in degrees

                    (oi, ri) = at.match_cxy(cat_ra,ref_ra,cat_dec,ref_dec,
                                      firstPass=initial, delta=final, log=log)

                    # If too few matches, assume the match was bad
                    if len(oi)<1:
                        oi = []

                    log.stdinfo("Matched %d objects in ['OBJCAT',%d] against"
                                " ['REFCAT',%d]" % (len(oi), extver, extver))

                    # Loop through the reference list updating the refid in the objcat
                    # and the refmag, if we can
                    for i in range(len(oi)):
                        objcat.data['REF_NUMBER'][oi[i]] = refcat.data['Id'][ri[i]]
                        if(magcolname):
                            objcat.data['REF_MAG'][oi[i]] = refcat.data[magcolname][ri[i]]
                            objcat.data['REF_MAG_ERR'][oi[i]] = refcat.data[magerrcolname][ri[i]]


            adoutput_list.append(ad)

        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


