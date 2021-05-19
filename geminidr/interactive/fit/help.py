
def _plot_tool(name, icon):
    return '<img width=16 height=16 src="dragons/static/help/' + icon + '.png"/><span><b>' + name + '</b></span><br/>'


tools_with_select = (
    ("Move", 'move', 'drag the plot around to reposition it'),
    ("Free-Select", 'lasso_select', 'draw an arbitrary area to select points for masking/unmasking<br/><i>also works with shift</i>'),
    ('Box Zoom', 'box_zoom', 'drag a rectangle to zoom into that area of the plot'),
    ('Box Select', 'box_select', 'drag a rectangle to select points for masking/unmasking<br/><i>also works with shift</i>'),
    ('Scroll Wheel Zoom', 'wheel_zoom', 'enable the scroll wheel to zoom the plot in or out'),
    ('Point Select', 'tap_select', 'click on individual points to select or unselect for masking/unmasking.<br/><i>also works with shift</i>'),
    ('Reset', 'reset', 'Reset the view, clearing any zoom or moves')
)

tools_without_select = (
    ("Move", 'move', 'drag the plot around to reposition it'),
    ('Box Zoom', 'box_zoom', 'drag a rectangle to zoom into that area of the plot'),
    ('Scroll Wheel Zoom', 'wheel_zoom', 'enable the scroll wheel to zoom the plot in or out'),
    ('Reset', 'reset', 'Reset the view, clearing any zoom or moves')
)

PLOT_TOOLS_WITH_SELECT_HELP_SUBTEXT = """
<h3>Plot Tools</h3>

<p>
<div class="plot_tools_help"><div>""" + \
    '\n'.join(_plot_tool(t[0], t[1]) for t in tools_with_select) + \
"""
</div></div>
Data points in the upper plot may be selected in order to mask or
unmask them from consideration.  To select, choose the <i>Box Select</i>, 
<i>Point Select</i>, or <i>Free-Select</i> tool to the right of the figure.  
Selections may be additive if you hold down the shift key.  Once you have a 
selection, you may <b>mask</b> or <b>unmask</b> the selection by hitting 
the <b>M</b> or <b>U</b> key respectively.
</p>
<dl>""" + \
    '\n'.join('<dt>' + t[0] + '</dt><dd>' + t[2] + '</dd>' for t in tools_with_select) + \
 """
</dl>
<br clear="all"/>"""


PLOT_TOOLS_HELP_SUBTEXT = """
<h3>Plot Tools</h3>

<p>
<div class="plot_tools_help"><div>""" + \
    '\n'.join(_plot_tool(t[0], t[1]) for t in tools_without_select) + \
"""
</div></div>
</p>
<dl>""" + \
    '\n'.join('<dt>' + t[0] + '</dt><dd>' + t[2] + '</dd>' for t in tools_without_select) + \
 """
</dl>
<br clear="all"/>"""


REGION_EDITING_HELP_SUBTEXT = """
<h3>Region Editing</h3>

<p>
In addition to the region edit text box, you can edit regions interactively
on the upper figure.  To start a new region, hover the mouse where you want
one edge of the region and hit <b>R</b>.  Then move to the other end of the
desired region and hit <b>R</b> again.  To edit an existing region, hit <b>E</b>
while the mouse is near the edge you wish to adjust.  To delete a region, hit
<b>D</b> while close to the region you want removed.  The <i>Regions</i>
text entry will also update with these changes and you may fine tune the 
results there as well.
</p>
"""


CALCULATE_SENSITIVITY_HELP_TEXT = """
<h2>Help</h2>
<p>
    Calculates the overall sensitivity of the observation system
    (instrument, telescope, detector, etc) for each wavelength using
    spectrophotometric data. It is obtained using the ratio
    between the observed data and the reference look-up data.</p>
<p>
    For that, it looks for reference data using the stripped and lower
    case name of the observed object inside geminidr.gemini.lookups,
    geminidr.core.lookups and inside the instrument lookup module.
</p>
<p>
    The reference data is fit using a Spline in order to match the input
    data sampling.
</p>
<h3>Profile parameters</h3>
<p>Those parameters applies to the computation of the 1D profile.</p>
<dl>
<dt>Order</dt>
<dd>
    Percentile to determine signal for each spatial pixel. Uses when
    collapsing along the dispersion direction to obtain a slit profile.
    If None, the mean is used instead.
</dd>
<dt>Sigma Clip, Upper, Lower</dt>
<dd>
    Enables sigma rejection with individually settable upper and lower
    sigma bounds.
</dd>
<dt>Max Iterations</dt>
<dd>
    Maximum number of rejection iterations
</dd>
<dt>Grow</dt>
<dd>
    Radius to reject pixels adjacent to masked pixels of spline fit
</dd>
<dt>Regions</dt>
<dd>
    Comma-separated list of colon-separated pixel coordinate pairs
    indicating the region(s) over which the input data should be
    used. The first and last values can be blank, indicating to
    continue to the end of the data.
</dd>
</dl>
""" + PLOT_TOOLS_WITH_SELECT_HELP_SUBTEXT + REGION_EDITING_HELP_SUBTEXT


NORMALIZE_FLAT_HELP_TEXT = """
<h2>Help</h2>
<p>
        This primitive normalizes a GMOS Longslit spectroscopic flatfield
        in a manner similar to that performed by gsflat in Gemini-IRAF.
        A cubic spline is fitted along the dispersion direction of each
        row, separately for each CCD.
</p><p>
        As this primitive is GMOS-specific, we know the dispersion direction
        will be along the rows, and there will be 3 CCDs.
</p><p>
        For Hamamatsu CCDs, the 21 unbinned columns at each CCD edge are
        masked out, following the procedure in gsflat.
</p>
<h3>Normalize parameters</h3>
<p>Those parameters applies to the computation of the 1D profile.</p>
<dl>
<dt>Order</dt>
<dd>
    Percentile to determine signal for each spatial pixel. Uses when
    collapsing along the dispersion direction to obtain a slit profile.
    If None, the mean is used instead.
</dd>
<dt>Sigma Clip, Upper, Lower</dt>
<dd>
    Enables sigma rejection with individually settable upper and lower
    sigma bounds.
</dd>
<dt>Max Iterations</dt>
<dd>
    Maximum number of rejection iterations
</dd>
<dt>Grow</dt>
<dd>
    Radius to reject pixels adjacent to masked pixels of spline fit
</dd>
<dt>Regions</dt>
<dd>
    Comma-separated list of colon-separated pixel coordinate pairs
    indicating the region(s) over which the input data should be
    used. The first and last values can be blank, indicating to
    continue to the end of the data.
</dd>
</dl>
""" + PLOT_TOOLS_HELP_SUBTEXT + REGION_EDITING_HELP_SUBTEXT


DEFAULT_HELP = """

<h2>Help</h2>

<p>DRAGONS Interactive Tools provide an interactive web interface to adjust
the data reduction parameters with a preview of the results.  This system
runs in your browser using a local bokeh server.  Adjust the parameters
to your liking and click the <b>Accept</b> button to continue.</p>
"""

TRACE_APERTURES = """
    <h1>Help for traceApertures</h1>

    <p> Traces the spectrum in 2D spectral images for each aperture center 
        stored in the APERTURE on each extension. The tracing starts at the 
        brightest region, instead of one of the edges of the detector. This 
        allows overcoming issues with sources with (almost) no flux in the 
        blue/red extremes. </p> 

    <p> The leftmost panel, named Tracing Parameters, in the Web User Interface 
        contains parameters used to perform the tracing, i.e., to follow how the 
        position in the spatial direction of our target changes along the 
        spectral direction. The Tracing Parameters are applied to all 
        appertures. </p>

    <p> You can find the traced data in the top plot where X represents the 
        pixels in the spectral direction and Y the pixels in the spatial 
        direction. Each traced data is represented by a black circle. </p>

    <p> You can perform a new tracing by changing the Tracing Parameters and 
        by clicking on the Trace Apertures button. Tracing can take a few 
        seconds deppending on your input parameters. If tracing fails, your will 
        receive an error message and the Web UI will return to the previous 
        working state. </p>

    <p> The red line in the top plot shows the function that better represents 
        how our target's spatial position varies continuously along the 
        dispersion direction, following the traced data using a Chebyshev 
        function. <p>  

    <p> You can change the parameters in the rightmost column within each tab, 
        which contains the Fitting Parameters for each APERTURE. If you change 
        a parameter, this primitive will fit again using the most recent parameters 
        and update the line in the plot area. </p>

    <p> For both Tracing Parameters and Fitting Parameters, you will find a 
        reset button. Each reset button only resets the parameters in the same 
        column it belongs. </p>        

    <p> Once you are satisfied with the tracing and the fit, press the Accept 
        button at the top right to continue your data reduction using the 
        parameters on screen. </p>

    <h2> Tracing Parameters: </h2>
    <dl>
        <dt> Max Missed </dt>
        <dd> Maximum number of steps to miss before a line is lost. </dd> 

        <dt> Max Shifted </dt>
        <dd> Maximum shift per pixel in line position. </dd>

        <dt> Number of Lines to Sum </dt>
        <dd> Number of lines to sum. </dd> 

        <dt> Tracing Step </dt>
        <dd> Step in rows/columns for tracing. </dd>
    </dl>

    <h2> Fitting Parameters </h2>
    <dl> 
        <dt> Function </dt>
        <dd> This is the Function used to fit the traced data. For this 
        primitive, we use Chebyshev. </dd>

        <dt> Order </dt>
        <dd> Order of Chebyshev function. </dd> 

        <dt> Max Iteractions </dt>
        <dd> Maximum number of rejection iterations. You can skip data rejection
             if you set this parameter to Zero. </dd>

        <dt> Sigma Clip </dt> 
        <dd> Reject outliers using sigma-clip. </dd>

        <dt> Sigma (Lower) </dt>
        <dd> Number of sigma used as lower threshold. </dd>

        <dt> Sigma (Upper) </dt>
        <dd> Number of sigma used as upper threshold/ </dd>

        <dt> Grow </dt> 
        <dd> If a point is rejected, then any points within a distance grow get 
             rejected too. </dd>

    </dl>
    """