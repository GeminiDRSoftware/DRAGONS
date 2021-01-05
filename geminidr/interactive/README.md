# Interactive Tools

This is a guide to using the interactive tools for data reduction.
For information on developing new interactive interfaces, see
![INTERACTIVE.md](INTERACTIVE.md)

## Normalize Flat

The step to normalize flats can be done interactively.  This happens
as part of the `makeProcessedFlat` recipe.  To turn on interactive mode,
pass the `interactive_reduce` parameter.  Here is an example.

### Step 1: Get Data

For this example, you will need to download `N20200520S0141.fits`.  You
can do this in your browser by browsing to:

<https://archive.gemini.edu/file/N20200520S0141.fits>

Alternatively, on UNIX based systems you can use

```bash
curl -O https://archive.gemini.edu/file/N20200520S0141.fits
```

### Step 2: Call `reduce` With `interactive_reduce` Parameter

```bash
reduce -p interactive_reduce=True -p do_bias=False N20200520S0141.fits
```

This should select the `makeProcessedFlat` recipe.  The `do_bias` parameter
just drops that step so we can focus on the `normalizeFlat` interactivity.
The user interface should automatically start in your web browser and will
look as follows:

### Step 3a: Alter Inputs As Desired

![Normalize Flat Input Descriptions](docs/NormalizeFlatInputCallouts.png)

There are two sets of inputs.  The left-most are applied across all of the
CCDs.  This is the fit function (i.e. spline) to perform and a row of data to sample.
The fit is too expensive to preview on all of the data, so this allows for
a responsive interface.  The inputs to the fit are available per-extension
and on each tab.

### Step 3b: Alter Regions As Desired

The regions can be specified as well.  This narrows the points used as input
to the fit.  This can be done interactively, or via the text input below 
each plot.  Regions are also set per-CCD.

### Step 4: Submit Parameters

When you hit the *Submit* button, the UI will close and the reduction will
continue, using the inputs and fit function you specified.

### Step 5: Check Output

After finishing, the result of the reduction will be available as 
`N20200520S0141_flat.fits`

