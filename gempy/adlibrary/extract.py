#! /usr/bin/env python
import sys, os
import time

import numpy as np
import pyfits as pf
from matplotlib import pyplot as pl

import segmentation as seg
from ..library import gfit

from astrodata import AstroData
from astrodata import new_pyfits_version
from astrodata.utils import Lookups


# Load the timestamp keyword dictionary.
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")
def print_timing(func):
    def wrapper(*arg,**kargs):
        t1 = time.time()
        res = func(*arg,**kargs)
        t2 = time.time()
        print '%s took %0.3fs' % (func.func_name, (t2-t1))
        return res
    return wrapper

def trace_footprints(ad, function='polynomial', order=2, 
                         trace_threshold=1., debug=False):
    """

    This function finds the footprints edges of spectroscopic flats, creates a 
    BINTABLE extension with the footprint parameters and appends it to the output
    AstroData object.

    :param ad: Input Astrodata object. 
    :param function: Name of the fitting function to use.
    :type function: Default is 'polynomial'.
    :param order: Degree of the polynomial.
    :type order: Default is 2
    :param trace_threshold: Threshold in units of sigma to applied to the
                            filtered image.
    :type trace_threshold: Default is 1.
    :param debug:  For debugging purposes.
    :return adoutput: Output Astrodata object containing the input ad object plus
                      the TRACEFP binary table.

    """

#try:
    # Find edges in the image, pair them in spectrum edges.
    footprints = find_footprints(ad, function=function, order=order,
                             trace_threshold=trace_threshold,debug=debug)

    # Create a FootprintTrace object.
    ft = FootprintTrace(footprints)

    # Use the footprint information to prepare BINTABLE format.
    # tb_adout is of type AD
    tb_adout = ft.as_bintable()
        
    # Append to the input AD object
    ad.append(tb_adout)

#except:
#    raise SystemError(repr(sys.exc_info()[1]))

    return ad


def find_footprints(ad, function='polynomial', order=2, trace_threshold=1.,
		debug=False):
    """ 
      Function to find footprint edges in an image that has
      been processed with the Sobel convolution kernel. 

      These are the steps to find the footprint:
      1)  Use the edge_detector_data() function to read the input data
          and MDF metadata from the AstroData object.
      2)  The function in 1) returns an <instrument>_EdgeDetector 
          subclass object.
      3)  Use this class method find_edges() returning two lists of lists,
          one is the list of (x,y) coordinates for all the left/bottom edges
          in the image and the other is the list of all the right/top edges
          in each spectrum.


      :: 

       Input
         ad:       AstroData object.
         function: Name of the fitting function to use.
         order:    Degree of the polynomial.
         trace_threshold: 
                   Threshold in units of sigma to applied to the
                   filtered image.

       Output: List of Footprint objects where each object contains: 
         id:     Integer reference number for footprint
         region: Section of the image where the footprint solution is 
                 valid, (x1, x2, y1, y2)   
         edges:  Tuple of Edge object (edge_1,edge_2) defining the 
                 long edges of the footprint.
         width:  Average width of the footprint (calculated with the edges)

    """

    
    # After setting metadata instantiate an EdgeDetector
    # class object, returning the object as the sublclass
    # instance to which the 'ad' object indicates.

    edt= seg.EdgeDetector(ad)

    # Get a list of lists with (x,y) for each edge.
    (xylist_1,xylist_2) = edt.find_edges()

    #  --- Create 2 lists of Edge objects. One (edge_1)
    #      has the left/bottom edge of the footprints and
    #      edge_2 has the right/top edge of the footprints.

    # get orientation in degress
    orientation = [0, 90][edt.axis]

    # initialize list of lef/bottom (edge_1) edges and
    # right/top (edge_2) edges. 
    edges_1 = []
    edges_2 = []

    # the lists xylist_1 and _2 are list of lists:
    # xylist_ = [[] for _ in range(len(reference_edges))]

    k=0
    for xy_1,xy_2 in zip(xylist_1,xylist_2):
        
        low = edt.footprint_low[k]
        high = edt.footprint_high[k]

        # For the current footprint (has 2 edges)
        # get the list of x's and y's.
        xx1 = np.asarray([x for x,y in xy_1])
        yy1 = np.asarray([y for x,y in xy_1])
        xx2 = np.asarray([x for x,y in xy_2])
        yy2 = np.asarray([y for x,y in xy_2])

        # For vertical edges, the yy1,yy2 are the independent
        # variable arrays. Instantiate the Edge class.
        
        # Instantiates left and right Edge objects
        ed1 = seg.Edge(xx1,yy1)
        ed2 = seg.Edge(xx2,yy2)
        ed1.orientation = orientation
        ed2.orientation = orientation

        # Set the low and high location in the dispersion 
        # direction. Fit edges.
        set_dispersion_limits(low,high,ed1,ed2,function,order)    

        k += 1

        # add to the current lists of edges.
        edges_1.append(ed1)
        edges_2.append(ed2)


    # --- Now that we have both footprint edges, make a list of
    #     Footprint objects.

    footprints = []
    sn = 1     
    for ed1,ed2 in zip(edges_1,edges_2):
        footprint = Footprint(sn, ed1, ed2)
        footprints.append(footprint)
        sn += 1

    if debug:
        _plot_footprints(edt.image, footprints)

    return footprints

def set_dispersion_limits(low,high,ed1,ed2,function,order):
    """ Dispersion limits (low,high) are the minimum and
        maximum value in the dispersion direction. They 
        are calculated using the instrument parameters
        like gratings and filters plus tables in 
        dictionaries and others. Please see footprint_len()
        for instrument specific details.

        If low and high are defined, set them as values in the
        Edge.trace (xx,yy) members and recalculate the
        fit function.
    """
    # Change the function and/or order if they are not the defaults.
    if (function !='polynomial') | (order != 2):
        for ed1,ed2 in zip(edges_1,edges_2):
            ed1.setfunction(function)
            ed1.setorder(order)
            ed2.setfunction(function)
            ed2.setorder(order)

    for ed in [ed1,ed2]:
        xx,yy = ed.trace

        # fit Edges
        ed.fitfunction() 

        # Make sure we have points in the range (iy1,iy2) in dispersion
        if ed.orientation == 90:
            g = np.where((yy >= low) & (yy <= high))
            xx,yy = (xx[g],yy[g])

            # Evaluate the spatial coordinate low and high
            xx[0],xx[-1] = ed.evalfunction([low,high])
            yy[0],yy[-1] = (low,high)
        else:
	    g = np.where((xx >= low) & (xx <= high))
    
	    # Evaluate the spatial coordinate low and high
	    xx[0],xx[-1] = (low,high)
	    yy[0],yy[-1] = ed.evalfunction([low,high])
    
        # Reset the trace attribute with these update coordinates.
        ed.trace = (xx,yy)

        # Now that we have good extreme points, fit again to set
        # xlim,ylim as well.
        ed.fitfunction()

    return

def _plot_footprints(image,footprints):
        """ 
          NOTE: This for development. Not for
                public release
          Plot the edges in the list array self.trace

            Plot the footprint edges using the fitting functions.
        """
        try:
            from stsci.numdisplay import display
        except ImportError:
            from numdisplay import display

        orientation = footprints[0].edges[0].orientation
 
        pl.clf()
        med = np.median(np.where(image>0,image,0))
        for k,footprint in enumerate(footprints):
            edge1 = footprint.edges[0]; edge2 = footprint.edges[1]
            xx1,yy1 = np.asarray(edge1.trace,int)
            xx2,yy2 = np.asarray(edge2.trace,int)
            # Plot (x_array,y_array)
            evf1 = edge1.evalfunction 
            evf2 = edge2.evalfunction 

            if orientation == 90:
                xx1 = np.asarray(evf1(yy1),int)
                xx2 = np.asarray(evf2(yy2),int)
                pl.plot(xx1,yy1,'b',xx2,yy2,'r')
            else:
                yy1 = np.asarray(evf1(xx1),int)
                yy2 = np.asarray(evf2(xx2),int)
                pl.plot(xx1,yy1,'b',xx2,yy2,'r')

            image[yy1,xx1]=med*2
            image[yy2,xx2]=med*2

        display(image,frame=2,quiet=True)
            
def _fit_edges(edges):
    """
      Fit a function to each edge.
      
      Input: 
         edges: List of Edge object with (x,y) traces
      
      returns:
         edges: Same as input plus fit coefficients for each edge.

    """
    for edge in edges:
        # Use default edge.function, edge.order 
        edge.fitfunction()

class Footprint(object):
    """
     Provides facilities to create a footprint each containing a pair
     of edge traces.

     ::

      Footprint attributes:

       id:     Integer reference number for footprint
       region: Section of the image where the footprint solution is valid,
               (x1, x2, y1, y2)   
       edges:  Tuple of Edge object (edge_1,edge_2) defining the 
               long edges of the footprint.
       width:  Average width of the footprint. 

    """

    def __init__(self, id, edge_1,edge_2):
    
        self.id = id
        self.edges = (edge_1,edge_2)
        x1 = np.min(edge_1.xlim+edge_2.xlim)
        x2 = np.max(edge_1.xlim+edge_2.xlim)
        y1 = np.min(edge_1.ylim+edge_2.ylim)
        y2 = np.max(edge_1.ylim+edge_2.ylim)

        if edge_1.orientation == 90:
            # Evaluate x2 with the fitting function using the
            # largest of y2 from both edges.
            x2 = edge_2.evalfunction(y2)[0]
            self.width = edge_2.evalfunction(y1)-edge_1.evalfunction(y1)
        else:
            # Evaluate y2 with the fitting function using the
            # largest of x2 from both edges.
            y2 = edge_2.evalfunction(x2)[0]
            self.width = edge_2.evalfunction(x1)-edge_1.evalfunction(x1)
            
        self.region = (x1, x2, y1, y2)


class FootprintTrace(object):
    """
      FootprintTrace provides facilities to create a BINTABLE
      extension with the input footprint list of objects.
     
      Attributes:

          footprints: Footprint object list.

      Methods:
          as_bintable:  Creates BINTABLE
    """

    def __init__(self,footprints):
        self.footprints = footprints

    def as_bintable(self):
        """
        Creates a BINTABLE object from the 
        FootprintTrace object.
        
        Input:
           self.footprints: list of Footprint objects.

        Output:
           AD: HDU astrodata object with a TRACEFP bintable extension.

        **Column discription**
        
        ::

         'id'       : integer reference number for footprint.
         'region'   : (x1,x2,y1,y2), window of pixel co-ords enclosing this
                      footprint. The origin of these coordinates could be 
                      the lower left of the original image.
         'range1'   : (x1,x2,y1,y2), range where edge_1 is valid.
                      The origin of these coordinates is the lower left of the
                      original image.
         'function1': Fit function name (default: polynomial) fitting edge_1.
         'coeff1'   : Arrray of coefficients, high to low order, such that
                      pol(x) = c1*x**2 + c2*x + c3   (for order 2).
         'order1'   : Order or polynomial (default: 2).
         'range2'   : ditto for edge_2.
         'function2': ditto for edges_2
         'coeff2'   : ditto for edges_2
         'order2'   : ditto for edges_2

         'cutrange1'   : (x1,x2,y1,y2), range where edge_1 is valid.
                         The origin of these coordinates is the lower left
                         of the cutout region.
         'cutfunction1': Fit function name (default: polynomial).
         'cutcoeff1'   : Arrray of coefficients, high to low order, such that
                         pol(x) = c1*x**2 + c2*x + c3   (for order 2)
         'cutorder1'   : Order or polynomial (default: 2).
         'cutrange2'   : ditto for edge_2
         'cutfunction2': ditto for edge_2
         'cutcoeff2'   : ditto for edge_2
         'cutorder2'   : ditto for edge_2

        """
        footprints = self.footprints

        # Get n_coeffs'. We are assuming they are the same for all edges.
        n_coeff = len(footprints[0].edges[0].coefficients)
        c1 = pf.Column (name='id',format='J')
        c2 = pf.Column (name='region',format='4E')
        c3 = pf.Column (name='range1',format='4E')
        c4 = pf.Column (name='function1',format='15A')
        c5 = pf.Column (name='order1',format='J')
        c6 = pf.Column (name='coeff1',format='%dE'%n_coeff)

        c7 = pf.Column (name='range2',format='4E')
        c8 = pf.Column (name='function2',format='15A')
        c9 = pf.Column (name='order2',format='J')
        c10 = pf.Column (name='coeff2',format='%dE'%n_coeff)

        c11 = pf.Column (name='cutrange1',format='4E')
        c12 = pf.Column (name='cutfunction1',format='15A')
        c13 = pf.Column (name='cutorder1',format='J')
        c14 = pf.Column (name='cutcoeff1',format='%dE'%n_coeff)

        c15 = pf.Column (name='cutrange2',format='4E')
        c16 = pf.Column (name='cutfunction2',format='15A')
        c17 = pf.Column (name='cutorder2',format='J')
        c18 = pf.Column (name='cutcoeff2',format='%dE'%n_coeff)

        nrows = len(footprints)
        tbhdu = pf.new_table(pf.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,\
                            c11,c12,c13,c14,c15,c16,c17,c18]),nrows=nrows)
        tb = tbhdu    # an alias

        # Write data to table columns

        orientation = footprints[0].edges[0].orientation
        for k,footprint in enumerate(footprints):
            edge1 = footprint.edges[0]; edge2 = footprint.edges[1]
            tb.data.field('id')[k]        = footprint.id
            tb.data.field('region')[k]    = np.asarray(footprint.region)

            # EGDE_1 DATA with respect to original image co-ords
            range1 = np.asarray(edge1.xlim+edge1.ylim)  # (x1, x2, y1, y2)
            tb.data.field('range1')[k]    = range1
            tb.data.field('function1')[k] = edge1.function
            tb.data.field('order1')[k]    = edge1.order
            tb.data.field('coeff1')[k]    = edge1.coefficients

            # EGDE_2 DATA with respect to original image co-ords
            range2 = np.asarray(edge2.xlim+edge2.ylim)  # (x1, x2, y1, y2)
            tb.data.field('range2')[k]    = range2
            tb.data.field('function2')[k] = edge2.function
            tb.data.field('order2')[k]    = edge2.order
            tb.data.field('coeff2')[k]    = edge2.coefficients

            region_x1 = footprint.region[0]
            region_y1 = footprint.region[2]
            # Setup the coefficient of the edge fit functions. We are
            # shifting the origin; so refit
            lcoeff=[]
            zval=[]
            for xx,yy in [edge1.trace,edge2.trace]:
                # We need to refit inside the cutregion
                xmr = xx - region_x1
                ymr = yy - region_y1
                if orientation == 0:
                    z = gfit.Gfit(xmr,ymr,edge1.function,edge1.order) 
                else:
                    z = gfit.Gfit(ymr,xmr,edge1.function,edge1.order) 
                lcoeff.append(z.coeff)
                zval.append(z)

            xlim1 = np.asarray(edge1.xlim)
            ylim1 = np.asarray(edge1.ylim)
            xlim2 = np.asarray(edge2.xlim)
            ylim2 = np.asarray(edge2.ylim)

            # Get the maximum values from both edges, so we can zero
            # the areas outside the footprint when cutting.
            #
            if orientation == 0:
                # Choose the largest x between both edges. 
                xmax = max(xlim1[1],xlim2[1])
                xlim1[1] = xmax
                xlim2[1] = xmax
                x1,x2 = (min(0,xlim1[0]),xmax)
                # And reevaluate the y values at this xmax
                y1 = ylim1[0] - region_y1
                y2 = zval[1](xmax)[0]
            else:
                # Choose the largest y between both edges
                ymax = max(ylim1[1],ylim2[1])
                ylim1[1] = ymax
                ylim2[1] = ymax
                y1,y2 = (min(0,ylim1[0]),ymax)
                # And reevaluate the x values at this ymax
                x1 = xlim1[0] - region_x1
                x2 = zval[1](ymax)[0] 

            # --- Set edge_1 data with respect to cutout image co-ords.
            tb.data.field('cutrange1')[k]    = (x1,x2,y1,y2)
            tb.data.field('cutfunction1')[k] = edge1.function
            tb.data.field('cutorder1')[k]    = edge1.order
            tb.data.field('cutcoeff1')[k]    = lcoeff[0]


            # --- Set edge_2 data with respect to cutout image co-ords
            # Applied offsets to range2 from footprint.region(x1,y1) 
            tb.data.field('cutrange2')[k]    = (x1,x2,y1,y2)
            tb.data.field('cutfunction2')[k] = edge2.function
            tb.data.field('cutorder2')[k]    = edge2.order
            tb.data.field('cutcoeff2')[k]    = lcoeff[1]

        # Add comment to TTYPE card
        hdr = tb.header
        if new_pyfits_version:
            hdr.update = hdr.set
        hdr.update('TTYPE2',hdr['TTYPE2'],
                  comment='(x1,y1,x2,y2): footprint window of pixel co-ords.')
        hdr.update('TTYPE3',hdr['TTYPE3'], comment='type of fitting function.')
        hdr.update('TTYPE4',hdr['TTYPE4'], comment='Number of coefficients.')
        hdr.update('TTYPE5',hdr['TTYPE5'], 
             comment='Coeff array: c[0]*x**3 + c[1]*x**2+ c[2]*x+c[3]')
        hdr.update('TTYPE6',hdr['TTYPE6'], 
             comment='(x1,y1,x2,y2): Edge fit window definition.')
        tb.header = hdr

        # Create an AD object with this
        tabad = AstroData(tbhdu)
        tabad.rename_ext("TRACEFP", 1)

        return tabad

    def _plot_footprints(self):
        """ 
          NOTE: This for development. Not for
                public release
          plot the edges in the list array self.trace
          i: sequential number, can be edge number

            Plot the footprint edges using the fitting functions.
        """
        footprints = self.footprints
        orientation = footprints[0].edges[0].orientation

        for k,footprint in enumerate(footprints):
            edge1 = footprint.edges[0]; edge2 = footprint.edges[1]
            xx1,yy1 = edge1.trace
            xx2,yy2 = edge2.trace
            evf1 = edge1.evalfunction 
            evf2 = edge2.evalfunction 
            if orientation == 0:
                pl.plot(xx1,evf1(xx1),'b',xx2,evf2(xx2),'r')
            else: 
                pl.plot(evf1(yy1),yy1,'b',evf2(yy2),yy2,'r')
             

import pywcs

def cut_footprints(ad, debug=False):
    """ Creates individual footprint images from the information in the 'TRACEFP'
        extension in the input AD object.
        It returns an AD object with a list of IMAGE extensions; each
        one containing a footprint cut.

        INPUT

           :param ad:    Astrodata object with a 'TRACEFP' bintable extension.
           :type adinput: Astrodata
           :param debug: Display and plot. If True it will display on DS9 the cuts
                         performed with the enclosed footprint. The footprints
                         will have the edges highlighted to show
                         how good the tracing was. It also shows
                         the cut and the edges on a plot.

        OUTPUT
           :return adout:  AD object with a list of IMAGE extensions containing 
                           one footprint per cut as describe in the TRACEFP bintable 
                           extension of the input AD object.

    """
    try:
        # Instantiate a CutFootprints object
        cl = CutFootprints(ad,debug)

        # Cut rectangular regions containing a footprint with its DQ and VAR 
        # region if any. The adout contains as many images extension as there
        # entries in the TRACEFP table.

        cl.cut_regions()
        adout = cl.as_astrodata()

    except:
        raise SystemError(repr(sys.exc_info()[1]))

    return adout


class CutFootprints():
    """ 
       CutFootprints provides facilites to build a list of footprint
       sections from the input TRACEFP table in the Astrodata object.

       :: 

        Methods
        -------
        
        cut_regions
            Loop through the records of the TRACEFP table creating
            one tuple per iteration containing (region, data,dq,and var
            cut out pixel data)
        cut_out: 
            Cut a region enclosing a footprint. 
        _init_as_astrodata
            Initialize parameters to be used by as_astrodata.
        as_astrodata
            Form an hdu and each cutout and append it to adout.


        Members
        -------
          ad:   AD object containing the extension 'TRACEFP'
                created by the ULF 'trace_footprint'.
          has_var,has_dq:
                Booleans to tell whether the VAR or DQ image sections 
                are in the input AD object.
          orientation:
                Int. Zero value for F2 and 90 for GMOS or GNIRS.
          filename,instrument: 
                Use for output information when debug is True.
          science,var,dq: 
                Reference to the input SCI, VAR and DQ image data.
          cut_list: 
                List with the tuples. Each contains region, sci_cut,
                var_cut and dq_cut footprint cut pixel data.
          wcs:  Input header WCS from pywcs.
                
          debug:Display and plot. If True it will display on DS9 the cuts
                performed with the enclosed footprint. The footprints
                will have the edges highlighted to show
                how good the tracing was. It also shows
                the cut and the edges on a plot.

               
    """

    def __init__(self,ad,debug=False):

        self.ad = ad
        self.has_dq = ad['DQ',1] != None
        self.has_var = ad['VAR',1] != None
        self.debug = debug
        instrument = ad.instrument()

        if instrument == 'F2':
            self.orientation = 90
        else:
            if ad['SCI',1].dispersion_axis() == 1:
                self.orientation = 0
            else:
                self.orientation = 90
        if debug:
            self.filename = ad.filename
            self.instrument = instrument

        self.dq = None
        self.var = None
        self.science =  self.ad['SCI',1].data
        if self.has_dq:
            self.dq   = self.ad['DQ', 1].data
        if self.has_var: 
            self.var  = self.ad['VAR',1].data

        self.cut_list = []        # Contains the list of Cut objects
        self.nregions = None        # The number of records in TRACEFP


    def cut_regions(self):
        """
            Loop through the records of the TRACEFP table creating
            one tuple per iteration with region,sci,dq,and var sections. 
            Then it appends each tuple to a list of cuts.

        """
        table = self.ad['TRACEFP'].data
        self.nregions = len(table.field('id'))

        
        if self.debug:
            plot_footprint(self.ad)

        for rec in table:
            cut_data = self.cut_out(rec)
            self.cut_list.append(cut_data)

    def cut_out(self,rec):
        """
          Cut a region enclosing a footprint. Each cut is defined by 'region'
          and the footprint in it is defined by the edges fitting functions.
          The science section is zero out between the rectangle borders
          and the footprint edge. The DQ section is bitwise ORed with 1.
          
          ::
  
           Input: 
             rec:     TRACEFP record
         
           Output:
             Tuple with region, sci_cut, dq_cut, var_cut data

        """ 
        # Input frames to get cut outs from
        science = self.science
        var     = self.var
        dq      = self.dq
        
        t1=time.time()
        id = rec.field('id')
        region = rec.field('region')
        cutrange1 = rec.field('cutrange1')
        cutrange2 = rec.field('cutrange2')
        evf1 = set_evalfunction(rec.field('cutcoeff1'))
        evf2 = set_evalfunction(rec.field('cutcoeff2'))

        # This is the rectangle coordinates containing one footprint
        rx1,rx2,ry1,ry2 = region
        
        # Get data sections. We add 1 to get the last element of the range.
        sci_cut = science[ry1:ry2+1,rx1:rx2+1].copy()

        # Define empty DQ, VAR cut array elements, in case the AD instance does 
        # not have them.
        dq_cut,var_cut=(None,None)
        has_dq = self.has_dq 
        has_var = self.has_var

        if has_dq:
            dq_cut = dq[ry1:ry2+1,rx1:rx2+1].copy()
        if has_var:
            var_cut = var[ry1:ry2+1,rx1:rx2+1].copy()

        # Now clear (zero out) the area of the data and dq cuts
        # between the rectangle and the footprint edges. 

        # Make indices representing the indices of a grid.
        y,x=np.indices(sci_cut.shape)

        # Generate mask values for the whole rectangular cut 
        # except for the footprint.
        if self.orientation == 90:
            # Mask values between indices of the cut left side
            # and the indices of the left edge.
            mask_1 = x < evf1(y)
            
            # Mask values between the right edge indices and
            # the indices of the cut right side.
            # index of the left edge
            mask_2 = evf2(y) < x
            mask = mask_1 + mask_2
        else:
            # bottom
            mask_1 = y < evf1(x) 
            # top
            mask_2 = evf2(x) < y
            mask = mask_1 + mask_2

        sci_cut[mask] = 0
        if has_var: var_cut[mask] = 0
        if has_dq: dq_cut[mask] = np.bitwise_or(dq_cut[mask],1)

        if self.debug:
            # plot and display
            plot_footprint_cut(sci_cut,x,y,self.orientation,evf1,evf2,
                         region, self.filename,self.instrument)
                          
        return (region, sci_cut, var_cut, dq_cut)

    def _init_as_astrodata(self):
        """
           Initialize parameters to be used by as_astrodata.

           Creates a WCS object (pywcs) from the SCI header and
           form the output AD object with the PHU and MDF from
           the input AD. We are adding the TRACEFP extension as well 
           for later use on the spectral reduction process. 

           Input:
              self.ad:  AD object.
           Output:
              adout:  Output AD object with AD phu and MDF 
        """

        ad = self.ad
        # Start output AD with the original phu and the MDF extension.
        adout = AstroData(phu=ad.phu)
        adout.append(ad['MDF'])
        adout.append(ad['TRACEFP'])

        # Get wcs information. It is in the PHU
        try:
            self.wcs = pywcs.WCS(ad.phu.header)
            if not hasattr(self.wcs.wcs, 'cd'):
                self.wcs = None
        except:   # Something wrong with WCS, set it to None
            self.wcs = None

        return adout
    
    def as_astrodata(self):
        """
            
          With each cut object in the cut_list having the SCI,DQ,VAR set,
          form an hdu and append it to adout.  Update keywords EXTNAME= 'SCI', 
          EXTVER=<footprint#>, CCDSEC, DISPAXIS, CUTSECT, CUTORDER in the header
          and reset WCS information if there was a WCS in the input AD header.

          ::

           Input:
              self.cut_list: List of Cut objects.
              self.adout:    Output AD object with MDF and
                             TRACEFP extensions.
           Output:
              adout: contains the appended HDUs.
        """

        adout = self._init_as_astrodata()

        ad = self.ad
        scihdr =        ad['SCI',1].header.copy()
        if self.has_dq:
            dqheader =  ad['DQ', 1].header.copy()
        if self.has_var:
            varheader = ad['VAR',1].header.copy()

        # Update NSCIEXT keyword to represent the current number of cuts.
        if new_pyfits_version:
            adout.phu.header.update = adout.phu.header.set
        adout.phu.header.update('NSCIEXT',len(self.cut_list)) 

        # This is a function renaming when using Pyfits 3.1
        if new_pyfits_version:
            scihdr.update = scihdr.set
        extver = 1

        # Generate the cuts using the region's sci_cut,var_cut and
        # dq_cut
        for region,sci_cut,var_cut,dq_cut in self.cut_list: 
            rx1,rx2,ry1,ry2 = np.asarray(region) + 1   # To 1-based
            csec = '[%d:%d,%d:%d]'%(rx1,rx2,ry1,ry2)
            scihdr.update('NSCUTSEC',csec,
                          comment="Region extracted by 'cut_footprints'")
            scihdr.update('NSCUTSPC',extver,comment="Spectral order")
            form_extn_wcs(scihdr, self.wcs, region)
            new_sci_ext = AstroData(data=sci_cut,header=scihdr)
            new_sci_ext.rename_ext(name='SCI',ver=extver)
            adout.append(new_sci_ext)
            if self.has_dq:
                new_dq_ext = AstroData(data=dq_cut, header=dqheader)
                new_dq_ext.rename_ext(name='DQ',ver=extver)
                adout.append(new_dq_ext)
            if self.has_var:
                new_var_ext = AstroData(data=var_cut, header=varheader)
                new_var_ext.rename_ext(name='VAR',ver=extver)
                adout.append(new_var_ext)
            extver += 1

        return adout

def set_evalfunction(coeff):
    """
        Utility function to form a polynomial given
        a coefficients array. 

        Input:
           coeff:  coefficients array, no greater than
                   4 elements.
        Output:
           eval:   Evaluator function
    """
    terms = ['','*x', '*x**2', '*x**3','*x**4']
    func = 'lambda x:'

    cc = coeff[::-1]     # The coefficients are from high order to low
    for i in range(len(cc)):
        func = func + '+%g'%cc[i]+terms[i]

    evf = eval(func)

    return evf

def form_extn_wcs(scihdr, wcs, region):
    """
      Form wcs information for this cut and
      update the header. The original WCS information
      is used to calculate CRVAL1,2 of the center
      of the cut. The CD matrix is unchanged.
      We used PYWCS module for the transformations.

      Input:
        scihdr:  SCI header from the original FITS WCS
           wcs:  WCS instance of pywcs
        region:  coords of the cut.
    """
    if wcs == None: return    # Don't do anything if wcs is bad
        
    kwlist = ['equinox','ctype1','cunit1','crpix1','crval1','ctype2','cunit2',
             'crpix2','crval2']

    # Get the WCS keywords from the PHU
    pheader = wcs.to_header()

    rx1,rx2,ry1,ry2 = np.asarray(region) + 1   # To 1-based 

    # Calculate crpix,crval for this section middle point
    cpix1 = (rx2-rx1+1)/2.0+rx1
    cpix2 = (ry2-ry1+1)/2.0+ry1
    (pheader['crval1'],pheader['crval2']), = wcs.wcs_pix2sky([[cpix1,cpix2]],1)
    pheader['crpix1'] = cpix1-rx1+1    # Set origin of the section to (1,1)
    pheader['crpix2'] = cpix2-ry1+1
 
    if new_pyfits_version:
        scihdr.update = scihdr.set
    # Now form the WCS header
    for kw in kwlist:
        scihdr.update(kw,pheader[kw])

    # Now CD. PC are the pywcs names for the CD's
    scihdr.update('cd1_1',pheader['pc1_1'])
    scihdr.update('cd1_2',pheader['pc1_2'])
    scihdr.update('cd2_1',pheader['pc2_1'])
    scihdr.update('cd2_2',pheader['pc2_2'])
    
    return

def plot_footprint(ad):
    """ Plot and display the edges found by trace_footprints.
        This information is the TRACEFP bintable
        extension in the AD object.
    """
    
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    if type(ad) is list: ad=ad[0]
    
    if ad.instrument() == 'F2':
        orientation = 90
    else:
        if ad['SCI',1].dispersion_axis() == 1:
            orientation = 0
        else:
            orientation = 90

    pl.clf()
    tb = ad['TRACEFP'].data
    data = ad['SCI',1].data
   
    for rec in tb:
        region = rec.field('region')
        cutrange1 = rec.field('cutrange1')
        cutrange2 = rec.field('cutrange2')
        coeff1 = rec.field('cutcoeff1')
        coeff2 = rec.field('cutcoeff2')
        evf1 = set_evalfunction(coeff1)
        evf2 = set_evalfunction(coeff2)

        if orientation == 0:
            x1,x2,y1,y2 = region

            x = np.arange(x1,x2+1)
            pl.plot(x,evf1(x)+y1,'b')        # edge_1
            pl.plot(x,evf2(x)+y1,'r')        # edge_2
            for x in np.arange(int(x1),int(x2+1.),20):
                xi = slice(x,x+5)
                xr = np.arange(x,x+5)
                yi = list(evf1(xr)+y1) + list(evf2(xr)+y1)
                data[yi,xi] = 9999
            
        else:    # Orientation is 90
            x1,x2,y1,y2 = region
            y = np.arange(y1,y2+1)
            pl.plot(evf1(y)+x1,y,'b')       # edge 1
            pl.plot(evf2(y)+x1,y,'r')       # edges 2
            off = evf1(y1)
            for y in range(int(y1),int(y2+1.),20):
                yi = slice(y,y+5)
                xi = list(evf1(np.arange(y,y+5))+x1-off) + \
                     list(evf2(np.arange(y,y+5))+x1-off)
                data[yi,xi] = 99999
                
    fname = os.path.basename(ad.filename)
    pl.title(fname+' ('+ad.instrument()+')')
    display(data,frame=1,quiet=True)
    time.sleep(3)

def plot_footprint_cut(data,x,y,orientation,evf1,evf2,region,filename,instru):
    """
       Plot the footprint cut inside a rectangle then
       display the same cut on frame 2.
    """
  
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    pl.clf()
    bval = data.max()
    bval += bval/10.
    #rx1,rx2,ry1,ry2 = region
   
    ny,nx = data.shape
    rxmin,rymin=(nx,ny)
    rxmax,rymax=(0,0)
    if (True):

        if orientation == 0:
            x = np.arange(nx)
            rxmin,rxmax = (0,nx)
            for evf,color in zip([evf1,evf2],['b','r']):
                zy = evf(x)
                pl.plot(x,zy,color)  
                imin = np.argmin(zy)
                imax = np.argmax(zy)
                rymin = min(rymin,zy[imin])
                rymax = max(rymax,zy[imax])
            pl.fill([rxmin,rxmax,rxmax,rxmin], [rymin,rymin,rymax,rymax],fill=False)
             
        else:    # Orientation is 90
            y = np.arange(ny)
            rymin,rymax = (0,ny)
            for evf,color in zip([evf1,evf2],['b','r']):
                zx = evf(y)
                pl.plot(zx,y,color)
                imin = np.argmin(zx)
                imax = np.argmax(zx)
                rxmin = min(rxmin,zx[imin])
                rxmax = max(rxmax,zx[imax])
            pl.fill([rxmin,rxmax,rxmax,rxmin], [rymin,rymin,rymax,rymax],fill=False)
    fname = os.path.basename(filename)
    pl.title(fname+' ('+instru+')')
    #pl.xlabel(str(region))
    pl.draw()
    display(data,frame=2,z1=0,z2=bval,quiet=True)
    time.sleep(1)

if __name__ == '__main__':
    """ Testing in the unix shell
    """
    from astrodata import AstroData
    import time
    
    f2='/data2/ed/data/fS20120104S0070.fits'
    gnirs = '/data2/ed/data/nN20101215S0475_comb.fits'
    gmos = '/data2/ed/data/mgS20100113S0110.fits'
    t1=time.time()


    for ff in [gmos,f2,gnirs]:
        ad = AstroData(ff)
        print 'MAIN:>>>>>>>>>>>>>>>>>',ad.instrument(),ad.filename
        adout = trace_footprints(ad,debug=False)
        print adout.info()
        t2 = time.time()
        print '.....trace_footprints:','(%.2f curr: %.1f)'%(t2-t1,t2-t1)
        cl = CutFootprints(adout,debug=False)
        t4=time.time()
        cl.cut_regions()
        t5=time.time()
        print '.....cut_regions:','(%.2f curr: %.1f)'%(t5-t4,t5-t1)
        adcut=cl.as_astrodata()
        t6=time.time()
        
        print '...cl.as_astrodata:','(%.2f curr: %.1f)'%(t6-t5,t6-t1)
        #adcut.filename='adcut.fits'
        #adcut.write(clobber=True)
        #raw_input('Enter...to continue')
       
    """
    k=1
    for ad in adcut:
       ad.filename='adlist'+str(k)+'.fits'
       print '...Writing', ad.filename
       ad.write(clobber=True)
       k=k+1
    print 'Total time:',time.time()-t1
    """


    
#nN20120305S0080_comb.fits   # GNIRS 4 files comb  (.15 pixscale)  [4 secs] (new:3.5)
#fS20120104S0070.fits        # F2 MOS     [21 secs]   (new: 8 sec)
#gS20100113S0110.fits    # prepare gmos MOS flat
#mgS20100113S0110.fits  # Mosaic GMOS         [58 secs] (new: 12 sec)


