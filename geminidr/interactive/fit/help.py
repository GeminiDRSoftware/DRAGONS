
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
