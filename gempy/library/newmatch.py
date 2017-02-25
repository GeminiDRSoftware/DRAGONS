import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.wcs import WCS
from scipy import optimize, spatial
from datetime import datetime
from gempy.utils import logutils
from geminidr.gemini.lookups import color_corrections


def match_coords(incoords, refcoords, radius=2.0, priority=[]):
    matched = np.full((len(incoords[0]),), -1, dtype=int)
    tree = spatial.cKDTree(zip(*refcoords))
    dist, idx = tree.query(zip(*incoords), distance_upper_bound=radius)
    for i in range(len(refcoords[0])):
        inidx = np.where(idx==i)[0][np.argsort(dist[np.where(idx==i)])]
        for ii in inidx:
            if ii in priority:
                matched[ii] = i
                break
        else:
            # No first_allowed so take the first one
            if inidx:
                matched[inidx[0]] = i
    return matched

def landstat(landscape, updated_model, x, y):
    xt, yt = updated_model(x, y)
    sum = np.sum([landscape[iy,ix] for ix,iy in zip((xt-0.5).astype(int),
                                                    (yt-0.5).astype(int))
                  if ix>=0 and iy>=0 and ix<landscape.shape[1]
                                     and iy<landscape.shape[0]])
    #print updated_model.parameters[0], updated_model.parameters[1], sum
    return -sum  # to minimize

def stat(ref_coords, updated_model, sigma, maxsig, x, y):
    f = 0.5/(sigma*sigma)
    maxsep = maxsig*sigma
    sum = 0.0
    xref, yref = ref_coords
    xt, yt = updated_model(x, y)
    for x, y in zip(xt, yt):
        sum += np.sum([np.exp(-f*((x-xr)*(x-xr)+(y-yr)*(y-yr)))
                       for xr, yr in zip(xref, yref)
                       if abs(x-xr)<maxsep and abs(y-yr)<maxsep])
    return -sum  # to minimize

class CatalogMatcher(Fitter):
    def __init__(self):
        super(CatalogMatcher, self).__init__(optimize.minimize,
                                             statistic=stat)

    def __call__(self, model, x, y, ref_coords, sigma=5.0, maxsig=5.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])
        farg = (model_copy, sigma, maxsig) +_convert_input(x, y, ref_coords)
        p0, _ = _model_to_fit_params(model_copy)
        result = self._opt_method(self.objective_function, p0, farg,
                                  **kwargs)
        fitted_params = result.x
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

class LandscapeFitter(Fitter):
    def __init__(self):
        super(LandscapeFitter, self).__init__(optimize.brute,
                                              statistic=landstat)

    def mklandscape(self, coords, sigma, maxsig, landshape):
        landscape = np.zeros(landshape)
        lysize, lxsize = landscape.shape
        hw = int(maxsig * sigma)
        xgrid, ygrid = np.mgrid[0:hw * 2 + 1, 0:hw * 2 + 1]
        rsq = (ygrid - hw) ** 2 + (xgrid - hw) ** 2
        mountain = np.exp(-0.5 * rsq / (sigma * sigma))
        for x, y in zip(*coords):
            mx1, mx2, my1, my2 = 0, hw * 2 + 1, 0, hw * 2 + 1
            lx1, lx2 = int(x - 0.5) - hw, int(x - 0.5) + hw + 1
            ly1, ly2 = int(y - 0.5) - hw, int(y - 0.5) + hw + 1
            if lx2 < 0 or lx1 >= lxsize or ly2 < 0 or ly1 >= lysize:
                continue
            if lx1 < 0:
                mx1 -= lx1
                lx1 = 0
            if lx2 > lxsize:
                mx2 -= (lx2 - lxsize)
                lx2 = lxsize
            if ly1 < 0:
                my1 -= ly1
                ly1 = 0
            if ly2 > lysize:
                my2 -= (ly2 - lysize)
                ly2 = lysize
            try:
                landscape[ly1:ly2, lx1:lx2] += mountain[my1:my2, mx1:mx2]
            except ValueError:
                print(y, x, landscape.shape)
                print(ly1, ly2, lx1, lx2)
                print(my1, my2, mx1, mx2)
        return landscape

    def __call__(self, model, x, y, ref_coords, sigma=5.0, maxsig=5.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])
        landscape = self.mklandscape(ref_coords, sigma, maxsig,
                                    (int(np.max(y)),int(np.max(x))))
        farg = (model_copy,) + _convert_input(x, y, landscape)
        p0, _ = _model_to_fit_params(model_copy)

        # TODO: Use the name of the parameter to infer the step size
        ranges = [slice(*(model_copy.bounds[p]+(sigma,)))
                  for p in model_copy.param_names]
        fitted_params = self._opt_method(self.objective_function,
                                         ranges, farg, finish=None, **kwargs)
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

def _new_match_objcat_refcat(ad):
    log = logutils.get_logger(__name__)

    filter_name = ad.filter_name(pretty=True)
    colterm_dict = color_corrections.colorTerms
    if filter_name in colterm_dict:
        formulae = colterm_dict[filter_name]
    else:
        log.warning("Filter {} is not in catalogs - will not be able to flux "
            "calibrate".format(filter_name))
        formulae = []

    try:
        refcat = ad.REFCAT
    except AttributeError:
        log.warning("No REFCAT present - cannot match to OBJCAT")
        return ad
    if not any(hasattr(ext, 'OBJCAT') for ext in ad):
        log.warning("No OBJCATs in {} - cannot match to REFCAT".
                    format(ad.filename))
        return ad

    # Try to be clever here, and work on the extension with the highest
    # number of matches first, as this will give the most reliable offsets.
    # which can then be used to constrain the other extensions. The problem
    # is we don't know how many matches we'll get until we do it, and that's
    # slow, so use OBJCAT length as a proxy.
    objcat_lengths = [len(ext.OBJCAT) if hasattr(ext, 'OBJCAT') else 0
                      for ext in ad]
    objcat_order = np.argsort(objcat_lengths)[::-1]

    pixscale = ad.pixel_scale()
    initial = 10.0/pixscale  # Search box size
    final = 1.0/pixscale     # Matching radius
    working_model = []

    for index in objcat_order:
        extver = ad[index].hdr['EXTVER']
        try:
            objcat = ad[index].OBJCAT
        except AttributeError:
            log.stdinfo('No OBJCAT in {}:{}'.format(ad.filename, extver))
            continue
        objcat_len = len(objcat)

        # The coordinates of the reference sources are corrected
        # to pixel positions using the WCS of the object frame
        wcs = WCS(ad.header[index+1])
        xref, yref = wcs.all_world2pix(refcat['RAJ2000'],
                                       refcat['DEJ2000'], 1)

        # Reduce the search radius if we've previously found a match
        if working_model:
            xref, yref = working_model[1](xref, yref)
            initial = 2.5/pixscale

        # First: estimate number of reference sources in field
        # Do better using actual size of illuminated field
        num_ref_sources = np.sum(np.all((xref>-initial,
            xref<ad[index].data.shape[1]+initial,
            yref>-initial,
            yref<ad[index].data.shape[0]+initial),
            axis=0))

        # How many objects do we want to try to match?
        if objcat_len > 2*num_ref_sources:
            keep_num = max(int(1.5*num_ref_sources),
                           min(10,objcat_len))
        else:
            keep_num = objcat_len
        sorted_idx = np.argsort(objcat['MAG_AUTO'])[:keep_num]
        xin, yin = objcat['X_IMAGE'][sorted_idx], objcat['Y_IMAGE'][sorted_idx]

        start = datetime.now()
        m_init = models.Shift(0.0) & models.Shift(0.0)
        fit_it = LandscapeFitter()
        m_init.offset_0.bounds = (-initial, initial)
        m_init.offset_1.bounds = (-initial, initial)
        ref_coords = (xref, yref)
        m = fit_it(m_init, xin, yin, ref_coords, sigma=10.0)
        print datetime.now() - start
        print m

        fit_it = CatalogMatcher()
        m_final = fit_it(m, xin, yin, ref_coords, method='Nelder-Mead')
        print m_final
        print datetime.now() - start

        xin, yin = objcat['X_IMAGE'], objcat['Y_IMAGE']
        matched = match_coords(m_final(xin, yin), ref_coords, sorted_idx, final)
        num_matched = sum(m>0 for m in matched)
        log.stdinfo("Matched {} objects in OBJCAT:{} against REFCAT".
                    format(num_matched, extver))
        # If this is a "good" match, save it
        if not working_model or num_matched > max(working_model[0], 2):
            working_model = (num_matched, m_final)

        # Loop through the reference list updating the refid in the objcat
        # and the refmag, if we can
        for i, m in enumerate(matched):
            if m > 0:
                objcat['REF_NUMBER'][i] = refcat['Id'][m]

                # Assign the magnitude
                if formulae:
                    mag, mag_err = _old_calculate_magnitude(formulae, refcat, matched[i])
                    objcat['REF_MAG'][i] = mag
                    objcat['REF_MAG_ERR'][i] = mag_err

    return ad

def _old_calculate_magnitude(formulae, refcat, indx):

    # This is a bit ugly: we want to iterate over formulae so we must
    # nest a single formula into a list
    if type(formulae[0]) is not list:
        formulae = [formulae]

    mags = []
    mag_errs = []
    for formula in formulae:
        mag = 0.0
        mag_err_sq = 0.0
        for term in formula:
            # single filter
            if type(term) is str:
                if term+'mag' in refcat.columns:
                    mag += refcat[term+'mag'][indx]
                    mag_err_sq += refcat[term+'mag_err'][indx]**2
                else:
                    # Will ensure this magnitude is not used
                    mag = np.nan
            # constant (with uncertainty)
            elif len(term) == 2:
                mag += float(term[0])
                mag_err_sq += float(term[1])**2
            # color term (factor, uncertainty, color)
            elif len(term) == 3:
                filters = term[2].split('-')
                if len(filters)==2 and np.all([f+'mag' in refcat.columns
                                               for f in filters]):
                    col = refcat[filters[0]+'mag'][indx] - \
                        refcat[filters[1]+'mag'][indx]
                    mag += float(term[0])*col
                    dmagsq = refcat[filters[0]+'mag_err'][indx]**2 + \
                        refcat[filters[1]+'mag_err'][indx]**2
                    # When adding a (H-K) color term, often H is a 95% upper limit
                    # If so, we can only return an upper limit, but we need to
                    # account for the uncertainty in K-band
                    if np.isnan(dmagsq):
                        mag -= 1.645*np.sqrt(mag_err_sq)
                    mag_err_sq += ((term[1]/term[0])**2 + dmagsq/col**2) * \
                        (float(term[0])*col)**2
                else:
                    mag = np.nan        # Only consider this if values are sensible
        if not np.isnan(mag):
            mags.append(mag)
            mag_errs.append(np.sqrt(mag_err_sq))

    # Take the value with the smallest uncertainty (NaN = large uncertainty)
    if mags:
        lowest = np.argmin(np.where(np.isnan(mag_errs),999,mag_errs))
        return mags[lowest], mag_errs[lowest]
    else:
        return -999, -999
