"""
The code in this module was originally written by Claude, guided by
Chris Simpson.

deripple.py -- measure and remove the etalon ripple from extracted echelle
               spectra (lamp flats, or telluric/science spectra divided by a
               smooth x telluric model) when the wavelength solution is only
               approximate (flexure, etc.).

Version 2 (2026-11-03) adds
  * --predict model|template : predict the ripple in orders where it is too weak to be
    detected, or which have too few usable pixels, from the orders where it is detected;
  * --extend (default on)    : evaluate the ripple on all 2048 pixels of every corrected
    order, including masked/NaN pixels (WAVE must give wavelengths for all pixels);
  * a continuum spline that tolerates gaps (masked absorption lines).
With predict=None and extend=False the results are identical to version 1.

Physics used
------------
The ripple is a fringe periodic in wavenumber sigma (cm^-1): a ~1 mm sapphire
plate, two birefringent components that beat.  Its fast phase is

      Phi(sigma) = 2 t n(lambda) sigma                 (cycles)

Everything uncertain -- a flexure shift of the spectrum, a small error in the
wavelength solution, a ppm change of the plate's thickness/temperature, the
beat between the two components, changes of fringe contrast -- changes the
phase/amplitude only *slowly* along an order.  Each order is therefore modelled
as
      ripple_i(x) = Re[ E_i(x) * exp(2 pi i Phi(sigma_i(x))) ]

with E_i(x) a low-order complex Legendre polynomial fitted independently in
every order (linear least squares).  A pixel shift becomes a constant phase in
E_i, a dispersion error a phase gradient; both are absorbed.

Carrier models (--carrier):
  sapphire (default)  Phi = 2 t n(lambda) sigma with the Sellmeier index of
                      sapphire (mean of o and e), t fixed at 1.0 mm (change
                      with --thickness MM).  Nothing is fitted globally, so it
                      is fast and robust even where the ripple is weak; an
                      error of up to ~1% in t is absorbed by the envelope.
                      --fit-thickness fits t instead (all orders together,
                      per-order phases free, search range --t-range).
  free                OPD(sigma) measured per order and fitted with a low-order
                      polynomial; use if the material is unknown.  Needs strong
                      ripple in several orders.

An order is corrected only if the ripple is detected: the fringe power at the
carrier frequency must stand out against the power at unrelated frequencies,
with S/N >= --snr-min (default 4) in the order itself, or S/N >= 2.5 in the
order and a combined S/N >= 6 for it and its two neighbours (the ripple
amplitude varies smoothly from order to order).

Steps: normalise orders (spline continuum, absorption lines masked) ->
fit carrier -> per-order detection + envelope fit -> iterate with a
ripple-aware continuum -> write outputs.
"""
import warnings
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import least_squares
from numpy.polynomial import legendre as L

from gempy.utils import logutils
log = logutils.get_logger(__name__)


NPIX = 2048
X = np.arange(NPIX, dtype=float)

# Sellmeier coefficients (lambda in µm): Malitson (1962) / Malitson & Dodge (1972)
_SAPPHIRE_O = ([1.4313493, 0.65054713, 5.3414021], [0.0726631, 0.1193242, 18.028251])
_SAPPHIRE_E = ([1.5039759, 0.55069141, 6.5927379], [0.0740288, 0.1216529, 20.072248])


def _sellmeier(coef, lam_um):
    """
    Evaluate a three-term Sellmeier equation.

    Parameters
    ----------
    coef : tuple of (sequence of float, sequence of float)
        Sellmeier coefficients ``(B, C)``, with ``C`` in µm, such that
        ``n^2 - 1 = sum_k B_k lam^2 / (lam^2 - C_k^2)``.
    lam_um : float or ndarray
        Vacuum wavelength in µm.

    Returns
    -------
    float or ndarray
        Refractive index at ``lam_um``.
    """
    B, C = coef; l2 = lam_um ** 2
    return np.sqrt(1 + sum(b * l2 / (l2 - c ** 2) for b, c in zip(B, C)))


def n_sapphire(sigma):
    """
    Refractive index of sapphire, averaged over the two polarizations.

    Parameters
    ----------
    sigma : float or ndarray
        Vacuum wavenumber in cm^-1.

    Returns
    -------
    float or ndarray
        Mean of the ordinary and extraordinary indices, ``(n_o + n_e) / 2``,
        from the Sellmeier fits of Malitson (1962) and Malitson & Dodge (1972).

    Notes
    -----
    The two birefringent fringe systems differ by ~0.4% in optical path; using
    the mean index places the carrier between them, and the difference (the
    beat) is absorbed by the per-order envelope.
    """
    lam = 1e4 / sigma
    return 0.5 * (_sellmeier(_SAPPHIRE_O, lam) + _sellmeier(_SAPPHIRE_E, lam))


def ng_sapphire(sigma, h=1.0):
    """
    Group index of sapphire (mean of the two polarizations).

    Parameters
    ----------
    sigma : float or ndarray
        Vacuum wavenumber in cm^-1.
    h : float, optional
        Step in cm^-1 for the central-difference derivative. Default 1.0.

    Returns
    -------
    float or ndarray
        ``n_g = n + sigma * dn/dsigma``.  The fringe spacing in wavenumber is
        ``1 / (2 t n_g)``, so ``2 t n_g`` is the (group) optical path difference.
    """
    return n_sapphire(sigma) + sigma * (n_sapphire(sigma + h) - n_sapphire(sigma - h)) / (2 * h)


def phi_reference(sigma):
    """
    Carrier phase of the reference (template) model: a 1.000 mm sapphire plate.

    Parameters
    ----------
    sigma : float or ndarray
        Vacuum wavenumber in cm^-1.

    Returns
    -------
    float or ndarray
        ``2 * 0.1 cm * n_sapphire(sigma) * sigma`` in cycles.
    """
    return 2 * 0.1 * n_sapphire(sigma) * sigma


# ------------------------------------------------- on-sky reference (template) models
# Two-fringe model, cubic P_j, Q_j (highest power first) as [P_1, Q_1, P_2, Q_2],
# u = (sigma - S0)/1000, phases relative to phi_reference().  Fitted to the frame-averaged
# order-centre envelopes of the IGRINS-2 telluric standards N20250701S0022,
# N20250717S0071/81/89, N20260201S0120/168 (A+B, divided by derippled flats).
# u_range: span of the order centres used in the fit; beyond it the model is held/continued
# (see `two_fringe`).
SKY_TEMPLATES = {
    'K': dict(S0=4450.0, deg=3, u_range=(-0.4558, 0.7578), p=np.array([
        1.160721, -0.520273, -1.92294, -4.35385, 0.16416, 0.273511, 2.294733, -1.235887,
        3.84883, -3.535312, -1.960329, -4.279933, -0.699217, -0.129429, 0.989804, 0.076165])),
    'H': dict(S0=6100.0, deg=3, u_range=(-0.6191, 0.7301), p=np.array([
        -20.637365, 7.367535, 8.081013, -8.611702, 6.705141, -7.575795, 4.269764, 4.180464,
        17.825158, -3.222918, -5.295856, -7.201106, -3.222481, -1.395816, 1.797185, -0.169751])),
}


# ----------------------------------------------------------------- utilities
def usable_mask(flux, frac=0.30):
    """
    Select the pixels of an order that are illuminated well enough to fit.

    Parameters
    ----------
    flux : ndarray, shape (NPIX,)
        Extracted flat spectrum of one order (NaN where unilluminated).
    frac : float, optional
        Minimum flux as a fraction of the order's 95th-percentile flux.
        Default 0.30; excludes the steep, poorly normalised order ends.

    Returns
    -------
    ndarray of bool, shape (NPIX,)
        True for usable pixels.  All False if fewer than 100 pixels are finite
        and positive.
    """
    g = np.isfinite(flux) & (flux > 0)
    if g.sum() < 100: return np.zeros_like(g)
    return g & (flux > frac * np.nanpercentile(flux[g], 95))


def fit_continuum(flux, good, knot=250, niter=5, ripple=None):
    """
    Fit a smooth continuum to one order, masking absorption lines.

    Parameters
    ----------
    flux : ndarray, shape (NPIX,)
        Extracted flat spectrum of one order.
    good : ndarray of bool, shape (NPIX,)
        Pixels to use (see `usable_mask`).
    knot : float, optional
        Interior knot spacing of the cubic spline in pixels.  Default 250,
        i.e. ~3 ripple periods, so the spline cannot follow the ripple.
    niter : int, optional
        Number of fit / line-masking iterations.  Default 5.
    ripple : ndarray, shape (NPIX,), optional
        Current fractional ripple model.  If given, ``flux / (1 + ripple)`` is
        fitted so that the continuum does not absorb any of the ripple.

    Returns
    -------
    cont : ndarray, shape (NPIX,)
        Continuum (NaN outside ``good``).
    keep : ndarray of bool, shape (NPIX,)
        ``good`` pixels that survived line masking.

    Notes
    -----
    Absorption lines are flagged as > 4 sigma (robust) below a 31-pixel running
    median of the residuals, plus gross outliers, and grown by +/-4 pixels.
    The knots are evenly spaced; if that leaves a knot interval without data
    (e.g. masked lines in a telluric spectrum), they are placed at quantiles of
    the usable pixels instead.
    """
    y = flux / (1 + ripple) if ripple is not None else flux
    xx = X[good]; yy = y[good]; ok = np.isfinite(yy)
    for _ in range(niter):
        nk = max(3, int(np.ptp(xx[ok]) / knot) + 1)
        t = np.linspace(xx[ok][0], xx[ok][-1], nk)[1:-1]
        try:
            s = LSQUnivariateSpline(xx[ok], yy[ok], t, k=3)
        except ValueError:
            t = np.unique(np.quantile(xx[ok], np.linspace(0, 1, nk)[1:-1]))
            s = LSQUnivariateSpline(xx[ok], yy[ok], t, k=3)
        res = yy / s(xx) - 1
        hp = res - median_filter(res, 31)
        sd = 1.4826 * np.median(np.abs(hp[ok]))
        bad = (hp < -4 * sd) | (np.abs(res) > 8 * max(np.std(res[ok]), 1e-4))
        ok = ~(np.convolve(bad, np.ones(9), 'same') > 0)
    cont = np.full(NPIX, np.nan); keep = np.zeros(NPIX, bool)
    cont[good] = s(xx); keep[good] = ok
    return cont, keep


_NULL_SCALES = np.r_[np.linspace(0.60, 0.90, 31), np.linspace(1.10, 1.50, 41)]


def detection_snr(phase, y, model=None):
    """
    Signal-to-noise ratio of the ripple in one order.

    Compares the power at the carrier phase with the mean power at unrelated
    carriers (the phase scaled by 0.6-0.9 and 1.1-1.5, i.e. several frequency
    resolution elements away), so non-white structure in the spectrum sets the
    noise level.

    Parameters
    ----------
    phase : ndarray
        Carrier phase in cycles at each pixel used.
    y : ndarray
        Normalised (and preferably high-passed, see `highpass`) flat values at
        the same pixels.
    model : ndarray, optional
        Fitted ripple at the same pixels.  If given, the noise power is
        measured on ``y - model`` so that leakage of a strong ripple into the
        reference frequencies does not cap the S/N.

    Returns
    -------
    float
        Approximately (ripple amplitude) / (its uncertainty).
    """
    p = phase - np.median(phase)
    P0 = np.abs(np.sum(y * np.exp(-2j * np.pi * p))) ** 2
    yn = y if model is None else y - model
    Pn = np.abs(np.exp(-2j * np.pi * np.outer(_NULL_SCALES, p)) @ yn) ** 2
    return np.sqrt(P0 / np.mean(Pn))


def envelope_fit(xn, phase, y, deg, nclip=3, clip=4.0, return_cov=False):
    """
    Fit a slowly varying complex envelope to a known fast carrier.

    Model: ``y = Re[ E(xn) * exp(2 pi i phase) ]`` with
    ``E(xn) = sum_k c_k P_k(xn)``, ``P_k`` Legendre polynomials and ``c_k``
    complex.  The model is linear in the real and imaginary parts of ``c_k``
    and is solved by least squares with iterative sigma clipping.

    Parameters
    ----------
    xn : ndarray
        Pixel coordinate scaled to [-1, 1].
    phase : ndarray
        Carrier phase in cycles.
    y : ndarray
        Normalised flat (fractional deviation from the continuum).
    deg : int
        Degree of the envelope polynomial.
    nclip : int, optional
        Number of clipping iterations.  Default 3.
    clip : float, optional
        Clipping threshold in robust standard deviations.  Default 4.
    return_cov : bool, optional
        Also return the covariance matrix of the real coefficient vector
        ``[Re c_0..c_deg, Im c_0..c_deg]``.  Default False.

    Returns
    -------
    c : ndarray of complex, shape (deg + 1,)
        Legendre coefficients of the envelope.
    amp : float
        Amplitude of a sinusoid with the same rms as the fitted model
        (fractional, not percent).
    w : ndarray of bool
        Pixels retained after clipping.
    cov : ndarray, shape (2 * (deg + 1), 2 * (deg + 1))
        Only if ``return_cov``: coefficient covariance (residual variance
        times ``(A^T A)^-1``).

    Notes
    -----
    A constant pixel shift of the spectrum changes the phase of ``E`` by a
    constant, and a small dispersion error adds a phase gradient; both are
    absorbed, so the fit is insensitive to the wavelength solution.
    """
    cosp, sinp = np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)
    V = L.legvander(xn, deg)
    A = np.hstack([V * cosp[:, None], -V * sinp[:, None]])
    w = np.ones(len(y), bool)
    for _ in range(nclip):
        co = np.linalg.lstsq(A[w], y[w], rcond=None)[0]
        r = y - A @ co
        w = np.abs(r) < clip * 1.4826 * np.median(np.abs(r[w]))
    c = co[:deg + 1] + 1j * co[deg + 1:]
    amp = np.sqrt(2) * np.std((A @ co)[w])      # rms-equivalent sinusoid amplitude
    if not return_cov:
        return c, amp, w
    s2 = np.sum(r[w] ** 2) / max(w.sum() - A.shape[1], 1)
    cov = s2 * np.linalg.pinv(A[w].T @ A[w])
    return c, amp, w, cov


# ------------------------------------------------------------- carrier fits
def highpass(y, k, width=40.):
    """
    High-pass filter an order by subtracting a masked Gaussian smoothing.

    Parameters
    ----------
    y : ndarray, shape (NPIX,)
        Normalised flat; values outside ``k`` are ignored (may be NaN-free
        placeholders).
    k : ndarray of bool, shape (NPIX,)
        Pixels to use in the smoothing.
    width : float, optional
        Gaussian sigma in pixels.  Default 40, which keeps >99% of a
        75-100 px ripple.

    Returns
    -------
    ndarray, shape (NPIX,)
        ``y`` minus its masked, normalised Gaussian smoothing.

    Notes
    -----
    Removes red (low-frequency) structure that would otherwise bias fringe
    frequency estimates and detection statistics.  Only used for searching and
    detection; the envelope fit uses the unfiltered data.
    """
    w = k.astype(float); v = np.where(k, y, 0.)
    return y - gaussian_filter1d(v, width) / np.maximum(gaussian_filter1d(w, width), 1e-3)


def _segments(norm, keep, sig):
    """
    Collect the usable, high-passed data of each order for carrier searches.

    Parameters
    ----------
    norm : ndarray, shape (nord, NPIX)
        Normalised flat (fractional deviation from the continuum).
    keep : ndarray of bool, shape (nord, NPIX)
        Usable pixels.
    sig : ndarray, shape (nord, NPIX)
        Wavenumber in cm^-1.

    Returns
    -------
    list of tuple
        ``(i, sigma, y)`` for every order ``i`` with at least 300 usable
        pixels; ``sigma`` and ``y`` are restricted to those pixels.
    """
    segs = []
    for i in range(len(norm)):
        k = keep[i]
        if k.sum() < 300: continue
        segs.append((i, sig[i][k], highpass(np.nan_to_num(norm[i]), k)[k]))
    return segs


def _semicoherent(segs, phase_fn, params):
    """
    Semi-coherent fringe power summed over orders.

    For each trial parameter, each order's data are projected onto a complex
    sinusoid with phase ``phase_fn(sigma, p)``; the captured fraction of the
    order's variance is summed over orders.

    Parameters
    ----------
    segs : list of tuple
        Output of `_segments`.
    phase_fn : callable
        ``phase_fn(sigma, p)`` returning the carrier phase in cycles.
    params : ndarray
        Trial values of the carrier parameter (e.g. thickness or OPD in cm).

    Returns
    -------
    ndarray, shape (len(params),)
        Summed fractional power; its maximum marks the best carrier.

    Notes
    -----
    The phase of each order is left free (powers, not amplitudes, are added),
    so the statistic is insensitive to order-to-order wavelength errors.
    """
    S = np.zeros(len(params))
    for i, s, y in segs:
        ph = np.array([phase_fn(s, p) - phase_fn(np.median(s), p) for p in params])
        P = np.abs(np.exp(-2j * np.pi * ph) @ y) ** 2
        S += 2 * P / (len(y) * np.sum(y ** 2))
    return S


def _peak(grid, S):
    """
    Location of the maximum of a sampled function, refined by a parabola.

    Parameters
    ----------
    grid : ndarray
        Uniformly spaced abscissae.
    S : ndarray
        Function values on ``grid``.

    Returns
    -------
    float
        Position of the maximum (the grid point itself if the maximum is at an
        end of the grid).
    """
    j = int(np.argmax(S))
    if 0 < j < len(S) - 1:
        d = 0.5 * (S[j - 1] - S[j + 1]) / (S[j - 1] - 2 * S[j] + S[j + 1])
        return grid[j] + d * (grid[1] - grid[0])
    return grid[j]


def fit_carrier_sapphire(segs, t_range_cm):
    """
    Fit the thickness of a sapphire plate to the ripple in all orders.

    The carrier phase is ``Phi(sigma) = 2 t n(sigma) sigma`` (cycles) with
    ``n`` from `n_sapphire`.  A coarse search (2 µm steps), whitened by a
    running median of the power, is followed by a fine search (0.1 µm steps,
    +/-30 µm) and parabolic peak refinement.

    Parameters
    ----------
    segs : list of tuple
        Output of `_segments`.
    t_range_cm : tuple of float
        ``(tmin, tmax)`` search range for the thickness in cm.

    Returns
    -------
    Phi : callable
        ``Phi(sigma)``: carrier phase in cycles for wavenumbers in cm^-1.
    info : dict
        ``t_mm`` : fitted thickness in mm.
        ``opd_mm_at`` : callable giving the group OPD ``2 t n_g`` in mm.

    Notes
    -----
    The whitening stops red (low-frequency) structure in the spectra from
    being mistaken for the fringe when the ripple is weak.
    """
    ph = lambda s, t: 2 * t * n_sapphire(s) * s
    t1 = np.arange(t_range_cm[0], t_range_cm[1], 0.0002)            # 2 µm steps
    S1 = _semicoherent(segs, ph, t1)
    # whiten: divide by a running median (±10% in t) so that red (low-frequency)
    # structure in the spectra cannot masquerade as the fringe
    bg = np.array([np.median(S1[np.abs(t1 / tt - 1) < 0.10]) for tt in t1])
    tc = t1[np.argmax(S1 / bg)]
    t2 = np.arange(tc - 0.003, tc + 0.003, 0.00001)                  # ±30 µm, 0.1 µm steps
    t = _peak(t2, _semicoherent(segs, ph, t2))
    return (lambda s: ph(s, t)), dict(t_mm=t * 10, opd_mm_at=lambda s: 2 * t * 10 * ng_sapphire(s))


# ------------------------------------------------- cross-order prediction
def two_fringe(p, u, deg, u_range=None):
    """
    Two-fringe envelope model across orders.

    Parameters
    ----------
    p : ndarray, shape (4 * (deg + 1),)
        Coefficients ``[P_1, Q_1, P_2, Q_2]``, each ``deg + 1`` values, highest
        power first.  ``P_j`` is the log-amplitude, ``Q_j`` the phase (cycles,
        relative to the carrier) of fringe system j.
    u : ndarray
        ``(sigma - S0) / 1000``.
    deg : int
        Polynomial degree.
    u_range : tuple of float, optional
        ``(umin, umax)`` range supported by data.  Outside it the amplitudes are
        held constant and the phases continued linearly (tangent at the end),
        so the model cannot run away when extrapolated.  Default: no limit.

    Returns
    -------
    ndarray of complex
        Envelope ``E(u) = sum_j exp(P_j(u)) exp(2 pi i Q_j(u))``.
    """
    n = deg + 1
    u = np.asarray(u, float)
    uc = u if u_range is None else np.clip(u, *u_range)
    out = 0
    for k in range(2):
        P = p[2 * n * k:2 * n * k + n]; Q = p[2 * n * k + n:2 * n * k + 2 * n]
        q = np.polyval(Q, uc)
        if u_range is not None:
            q = q + np.polyval(np.polyder(Q), uc) * (u - uc) if deg > 0 else q
        out = out + np.exp(np.polyval(P, uc)) * np.exp(2j * np.pi * q)
    return out


def _envelope_samples(fits, applied, sig, Phi, Phi_out, npts=5):
    """
    Sample the fitted envelopes of the detected orders for the cross-order fit.

    Returns arrays (sigma, E, err) where E is converted from the carrier ``Phi``
    used in the per-order fits to the carrier ``Phi_out`` of the prediction model.
    """
    S, E, e = [], [], []
    for i, (ph, xn, c, amp, w, cov, lo, hi) in fits.items():
        if not applied[i]: continue
        deg = len(c) - 1
        xs = np.linspace(-0.8, 0.8, npts)
        xpix = lo + (xs + 1) / 2 * (hi - lo)
        s = np.interp(xpix, X, sig[i])
        V = L.legvander(xs, deg)
        Ek = V @ c
        n = deg + 1
        var_re = np.einsum('ij,jk,ik->i', V, cov[:n, :n], V)
        var_im = np.einsum('ij,jk,ik->i', V, cov[n:, n:], V)
        Ek = Ek * np.exp(2j * np.pi * (Phi(s) - Phi_out(s)))
        S.append(s); E.append(Ek); e.append(np.sqrt(0.5 * (var_re + var_im)) * np.sqrt(npts / (deg + 1.)))
    if not S: return np.array([]), np.array([]), np.array([])
    return np.concatenate(S), np.concatenate(E), np.concatenate(e)


def predict_model(fits, applied, sig, Phi, mode, band=None, max_degree=3, start=None):
    """
    Fit the cross-order two-fringe model to the envelopes of the detected orders.

    Parameters
    ----------
    fits : dict
        Per-order fit results collected in `deripple`.
    applied : ndarray of bool
        Orders whose ripple was detected (only these constrain the model).
    sig : ndarray, shape (nord, NPIX)
        Wavenumbers (cm^-1) of every pixel.
    Phi : callable
        Carrier used in the per-order fits.
    mode : {'model', 'template'}
        See the module docstring.
    band : {'K', 'H'} or None
        Template band for ``mode='template'`` (default: inferred from sigma).
    max_degree : int
        Maximum polynomial degree for ``mode='model'``.
    start : ndarray, optional
        Starting parameters (e.g. from the previous iteration).

    Returns
    -------
    dict or None
        ``evaluate(sigma)`` -> ripple envelope times carrier, i.e. a callable
        giving the fractional ripple ``Re[E(sigma) exp(2 pi i Phi_out(sigma))]``;
        plus ``mode``, ``params``, ``deg``, ``S0``, ``chi2``, ``n_orders``,
        ``band``, ``scale``/``phase`` (template).  None if the model cannot be fitted.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        return _predict_model(fits, applied, sig, Phi, mode, band, max_degree, start)


def _predict_model(fits, applied, sig, Phi, mode, band, max_degree, start):
    n_det = int(np.sum([applied[i] for i in fits]))
    if mode == 'template':
        if band is None: band = 'K' if np.nanmedian(sig) < 5500 else 'H'
        T = SKY_TEMPLATES[band]
        s, E, e = _envelope_samples(fits, applied, sig, Phi, phi_reference)
        if len(s) == 0: return None
        u = (s - T['S0']) / 1000
        Eref = two_fringe(T['p'], u, T['deg'], T['u_range'])
        f = lambda q: np.r_[((q[0] * np.exp(2j * np.pi * q[1]) * Eref - E) / e).real,
                            ((q[0] * np.exp(2j * np.pi * q[1]) * Eref - E) / e).imag]
        best = None
        for q0 in ([start] if start is not None else []) + [np.array([1.0, k / 8.]) for k in range(8)]:
            r = least_squares(f, q0, loss='soft_l1', f_scale=3.0, x_scale='jac')
            if best is None or r.cost < best.cost: best = r
        q = best.x
        chi2 = np.sum(f(q) ** 2) / max(2 * len(s) - 2, 1)
        ev = lambda sig, q=q, T=T: np.real(q[0] * np.exp(2j * np.pi * q[1]) *
                                           two_fringe(T['p'], (sig - T['S0']) / 1000, T['deg'], T['u_range']) *
                                           np.exp(2j * np.pi * phi_reference(sig)))
        return dict(evaluate=ev, mode='template', band=band, params=q, scale=q[0],
                    phase=(q[1] + 0.5) % 1 - 0.5, deg=T['deg'], S0=T['S0'], chi2=chi2, n_orders=n_det)

    # mode == 'model'
    if n_det < 3: return None
    deg = int(min(max_degree, n_det - 2))
    s, E, e = _envelope_samples(fits, applied, sig, Phi, Phi)
    S0 = float(np.round(np.median(s), -1))
    u = (s - S0) / 1000
    urange = (u.min(), u.max())
    f = lambda p: np.r_[((two_fringe(p, u, deg) - E) / e).real, ((two_fringe(p, u, deg) - E) / e).imag]
    starts = []
    if start is not None and len(start) == 4 * (deg + 1):
        starts.append(start)                           # later iterations: start from the previous solution
    la = np.log(max(np.median(np.abs(E)), 1e-5) / 2 + 1e-6)
    z = [0.0] * deg
    grid = np.linspace(-0.8, 0.8, 9) if not starts else []   # grid search only without a previous solution
    for d1 in grid:
        for d2 in grid:
            if d2 <= d1: continue
            for ph1 in (0.0, 0.5):
                P = z + [la]
                starts.append(np.r_[P, [0.0] * (deg - 1), d1, ph1, P, [0.0] * (deg - 1), d2, 0.25])
    best = None
    for p0 in starts:
        r = least_squares(f, p0, loss='soft_l1', f_scale=3.0, x_scale='jac', max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    p = best.x
    chi2 = np.sum(f(p) ** 2) / max(2 * len(s) - len(p), 1)
    ev = lambda sig, p=p: np.real(two_fringe(p, (sig - S0) / 1000, deg, urange) * np.exp(2j * np.pi * Phi(sig)))
    return dict(evaluate=ev, mode='model', params=p, deg=deg, S0=S0, u_range=urange, chi2=chi2, n_orders=n_det)


# ----------------------------------------------------------------- main routine
def deripple(data, wave, first_order=None, t_range_mm=(0.8, 1.2),
             thickness_mm=1.0, envelope_degree=2, snr_min=4.0, snr_weak=2.5,
             snr_group=6.0, n_iter=3, predict=None, predict_degree=3,
             band=None, extend=True, verbose=True):
    """
    Measure and remove the etalon ripple from an extracted echelle flat (or a
    telluric/science spectrum divided by a smooth x telluric model).

    Parameters
    ----------
    data : array_like, shape (nord, NPIX)
        Extracted spectrum, one order per row, NaN where unilluminated or masked.
    wave : array_like, shape (nord, NPIX)
        Approximate wavelengths in nm, same shape as ``data``: row ``i``
        holds the wavelengths of row ``i`` of ``data``.  NaN is allowed.
    first_order : int, optional
        Echelle order number of row 0, used only to label the report.
        Default: label orders by row index (0, 1, 2, ...).
    carrier : {'sapphire', 'free'}, optional
        Carrier model.  ``'sapphire'`` (default) uses a sapphire plate of
        thickness ``thickness_mm``; ``'free'`` fits an empirical OPD(sigma)
        (see `fit_carrier_free`).
    t_range_mm : tuple of float, optional
        Thickness search range in mm, used only when ``carrier='sapphire'``
        and ``thickness_mm`` is None.  Default (0.8, 1.2).
    thickness_mm : float or None, optional
        Fixed sapphire thickness in mm.  Default 1.0.  None fits the
        thickness (see `fit_carrier_sapphire`).  Errors up to ~1% are
        absorbed by the envelope.
    opd_range_mm : tuple of float, optional
        OPD search range in mm for ``carrier='free'``.  Default (3.40, 3.75).
    envelope_degree : int, optional
        Degree of the per-order complex envelope polynomial.  Default 2.
    snr_min : float, optional
        Correct an order whose own ripple S/N is at least this.  Default 4.
    snr_weak : float, optional
        Also correct an order with S/N >= ``snr_weak`` if the combined S/N of
        it and its two neighbours (root sum of squares) is at least
        ``snr_group``.  Default 2.5.
    snr_group : float, optional
        See ``snr_weak``.  Default 6.
    n_iter : int, optional
        Number of continuum / ripple iterations.  Default 3.
    predict : {None, 'model', 'template'}, optional
        Predict the ripple in orders where it is not detected (or that have
        fewer than 300 usable pixels) from the detected orders, using the
        cross-order two-fringe model (see the module docstring).  None
        (default) leaves those orders uncorrected, as in version 1.
        ``'model'``: all model coefficients fitted to this spectrum (flats
        and tellurics; needs >= 3 detected orders).  ``'template'``: IGRINS-2
        on-sky reference shape, only scale and phase fitted (telluric/science
        spectra only; needs >= 1 detected order).
    predict_degree : int, optional
        Maximum polynomial degree of the ``'model'`` prediction.  Default 3;
        reduced automatically to (number of detected orders - 2).
    band : {'K', 'H'} or None, optional
        Band of the template for ``predict='template'``.  Default: inferred
        from the wavelengths (K if the median wavenumber < 5500 cm^-1).
    extend : bool, optional
        Evaluate the ripple on all NPIX pixels of every corrected or predicted
        order, including masked/NaN pixels and beyond the fitted range (the
        envelope is held constant beyond the fitted pixel range).  Requires
        wavelengths for all pixels (the ripple is left at zero where ``wave``
        is NaN).  Default True.
        False reproduces version 1 (ripple only where ``data`` is finite).
    verbose : bool, optional
        Print the carrier, the prediction model and a per-order report.
        Default True.

    Returns
    -------
    dict
        ``ripple`` : ndarray, shape (nord, NPIX)
            Fractional ripple ``r``; zero where not corrected.
        ``data_corrected`` : ndarray, shape (nord, NPIX)
            ``data / (1 + ripple)``.
        ``normalized`` : ndarray, shape (nord, NPIX)
            Flat divided by its continuum, minus 1.
        ``keep`` : ndarray of bool, shape (nord, NPIX)
            Pixels used in the fits.
        ``sigma`` : ndarray, shape (nord, NPIX)
            Wavenumber in cm^-1 (from the approximate wavelengths; NaN where
            the wavelength is NaN).
        ``orders`` : ndarray of int, shape (nord,)
            Echelle order (or row index) of each row.
        ``phase`` : callable
            Carrier phase ``Phi(sigma)`` in cycles.
        ``info`` : dict
            Carrier information (``t_mm``, ``opd_mm_at``, ``fixed``).
        ``report`` : ndarray, shape (nord, 7)
            Per order: order, amplitude [%], S/N, applied (0/1),
            rms before [%], rms after [%], predicted (0/1).  For predicted
            orders the amplitude is that of the predicted ripple.
        ``predicted`` : ndarray of bool, shape (nord,)
            Orders whose ripple comes from the cross-order prediction.
        ``prediction`` : dict or None
            Fitted prediction model (``mode``, ``params``, ``deg``, ``S0``,
            ``chi2``, ``n_orders``, ``evaluate``; template: ``band``,
            ``scale``, ``phase``), or None.

    Notes
    -----
    Each order is modelled as ``Re[E_i(x) exp(2 pi i Phi(sigma))]``:
    the carrier ``Phi`` carries the fast fringe phase, and the low-order
    complex envelope ``E_i`` absorbs flexure shifts, wavelength-solution
    errors, the beat between the two birefringent components and changes of
    fringe contrast.  No flexure correction or exact wavelength solution is
    therefore needed.

    Raises
    ------
    ValueError
        If ``data`` and ``wave`` do not have the same shape, or ``predict`` is
        not one of the allowed values.

    Examples
    --------
    >>> res = deripple(flat, wave, first_order=70, predict='model')
    >>> clean = res['flat_corrected']   # flat == clean * (1 + res['ripple'])
    """
    data = np.asarray(data, float); wave = np.asarray(wave, float)
    if data.shape != wave.shape:
        raise ValueError(f'data {data.shape} and wave {wave.shape} must have the same shape '
                         '(same orders, row for row)')
    if predict not in (None, 'model', 'template'):
        raise ValueError("predict must be None, 'model' or 'template'")
    nord = data.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = 1e7 / wave                                   # wavenumber, cm^-1
    orders = (first_order or 0) + np.arange(nord)
    s0 = np.nanmedian(sig)
    good = np.array([usable_mask(r) for r in data])
    ripple = np.zeros_like(data)
    pred = None; pstart = None
    for it in range(n_iter):
        norm = np.full_like(data, np.nan); keep = np.zeros_like(good)
        for i in range(nord):
            if good[i].sum() < 300: continue
            cont, k = fit_continuum(data[i], good[i], ripple=ripple[i] if it else None)
            norm[i] = data[i] / cont - 1
            keep[i] = k & np.isfinite(norm[i]) & np.isfinite(sig[i])
        segs = _segments(norm, keep, sig)
        if thickness_mm is None:      # fixed plate, nothing fitted
            Phi, info = fit_carrier_sapphire(segs, (t_range_mm[0] / 10, t_range_mm[1] / 10))
        else:
            t = thickness_mm / 10.
            Phi = lambda s, t=t: 2 * t * n_sapphire(s) * s
            info = dict(t_mm=thickness_mm, opd_mm_at=lambda s, t=t: 2 * t * 10 * ng_sapphire(s), fixed=True)
        ripple = np.zeros_like(data); fits = {}
        snrs = np.full(nord, np.nan)
        for i in range(nord):
            k = keep[i]
            if k.sum() < 300: continue
            ph = Phi(sig[i])
            lo, hi = X[k].min(), X[k].max()
            xn = np.clip(2 * (X - lo) / (hi - lo) - 1, -1, 1)   # envelope held constant beyond fitted range
            c, amp, w, cov = envelope_fit(xn[k], ph[k], norm[i][k], envelope_degree, return_cov=True)
            mod = np.real(L.legval(xn[k], c) * np.exp(2j * np.pi * ph[k]))
            snrs[i] = detection_snr(ph[k], highpass(np.nan_to_num(norm[i]), k)[k], mod)
            fits[i] = (ph, xn, c, amp, w, cov, lo, hi)
        # decision: S/N >= snr_min in the order itself, or a marginal detection
        # (S/N >= snr_weak) supported by the neighbouring orders (the ripple
        # amplitude varies smoothly from order to order)
        s2 = np.nan_to_num(snrs) ** 2
        group = np.sqrt(np.convolve(s2, np.ones(3), 'same'))
        applied = np.array([i in fits and (snrs[i] >= snr_min or (snrs[i] >= snr_weak and group[i] >= snr_group))
                            for i in range(nord)])
        for i in np.where(applied)[0]:
            ph, xn, c = fits[i][:3]
            fin = np.isfinite(ph) if extend else (np.isfinite(data[i]) & np.isfinite(ph))
            ripple[i][fin] = np.real(L.legval(xn[fin], c) * np.exp(2j * np.pi * ph[fin]))
        # prediction for the orders without a detected ripple
        predicted = np.zeros(nord, bool)
        if predict is not None:
            pred = predict_model(fits, applied, sig, Phi, predict, band=band,
                                 max_degree=predict_degree, start=pstart)
            if pred is None:
                if verbose and it == n_iter - 1:
                    warnings.warn(f'predict={predict!r}: too few detected orders, no prediction made')
            else:
                pstart = pred['params']
                for i in range(nord):
                    if applied[i]: continue
                    fin = np.isfinite(sig[i]) if extend else (np.isfinite(data[i]) & np.isfinite(sig[i]))
                    if not fin.any(): continue
                    ripple[i][fin] = pred['evaluate'](sig[i][fin])
                    predicted[i] = True
        report = []
        for i in range(nord):
            kk = keep[i]
            if predicted[i]:
                sel = kk if kk.sum() > 0 else (ripple[i] != 0)
                amp_p = np.sqrt(2) * np.std(ripple[i][sel]) * 100 if sel.any() else np.nan
                if i in fits:
                    wv = fits[i][4]
                    rms0 = np.std(norm[i][kk][wv]); rms1 = np.std((norm[i] - ripple[i])[kk][wv])
                    report.append((orders[i], amp_p, snrs[i], 0, rms0 * 100, rms1 * 100, 1))
                else:
                    report.append((orders[i], amp_p, np.nan, 0, np.nan, np.nan, 1))
                continue
            if i not in fits:
                report.append((orders[i], np.nan, np.nan, 0, np.nan, np.nan, 0)); continue
            ph, xn, c, amp, w = fits[i][:5]
            rms0 = np.std(norm[i][kk][w]); rms1 = np.std((norm[i] - ripple[i])[kk][w])
            report.append((orders[i], amp * 100, snrs[i], int(applied[i]), rms0 * 100, rms1 * 100, 0))
        if verbose:
            msg = f'iteration {it + 1}: OPD(σ={s0:.0f}) = {info["opd_mm_at"](s0):.4f} mm'
            if 't_mm' in info:
                msg += f'  (sapphire t = {info["t_mm"]:.4f} mm{", fixed" if info.get("fixed") else ""})'
            log.stdinfo(msg)
    report = np.array(report, float)
    if verbose:
        if pred is not None:
            if pred['mode'] == 'template':
                log.stdinfo(f'prediction: template {pred["band"]}, scale {pred["scale"]:.3f}, phase {pred["phase"]:+.4f} cycles, '
                      f'{pred["n_orders"]} detected orders, chi2/dof {pred["chi2"]:.2f}')
            else:
                log.stdinfo(f'prediction: two-fringe model, degree {pred["deg"]}, {pred["n_orders"]} detected orders, '
                      f'chi2/dof {pred["chi2"]:.2f}')
        log.stdinfo(' order  amp[%]   S/N  ripple     rms before -> after [%]')
        for r in report:
            how = 'predicted' if r[6] else ('measured ' if r[3] else 'none     ')
            log.stdinfo(f'{int(r[0]):6d}  {r[1]:6.3f} {r[2]:6.1f}  {how}  {r[4]:.3f} -> {r[5]:.3f}')
    return dict(ripple=ripple, data_corrected=data / (1 + ripple), normalized=norm, keep=keep,
                sigma=sig, orders=orders, phase=Phi, info=info, report=report,
                predicted=predicted, prediction=pred)


def old_deripple(flat, wave, first_order=None, carrier='sapphire',
             t_range_mm=(0.8, 1.2), thickness_mm=1.0, envelope_degree=2,
             snr_min=4.0, snr_weak=2.5, snr_group=6.0, n_iter=3):
    """
    Measure and remove the etalon ripple from an extracted echelle flat.

    Parameters
    ----------
    flat : array_like, shape (nord, NPIX)
        Extracted flat, one order per row, NaN where unilluminated.
    wave : array_like, shape (nord, NPIX)
        Approximate wavelengths in nm, same shape as ``flat``: row ``i``
        holds the wavelengths of row ``i`` of ``flat``.  NaN is allowed.
    first_order : int, optional
        Echelle order number of row 0, used only to label the report.
        Default: label orders by row index (0, 1, 2, ...).
    t_range_mm : tuple of float, optional
        Thickness search range in mm, used only when ``carrier='sapphire'``
        and ``thickness_mm`` is None.  Default (0.8, 1.2).
    thickness_mm : float or None, optional
        Fixed sapphire thickness in mm.  Default 1.0.  None fits the
        thickness (see `fit_carrier_sapphire`).  Errors up to ~1% are
        absorbed by the envelope.
    envelope_degree : int, optional
        Degree of the per-order complex envelope polynomial.  Default 2.
    snr_min : float, optional
        Correct an order whose own ripple S/N is at least this.  Default 4.
    snr_weak : float, optional
        Also correct an order with S/N >= ``snr_weak`` if the combined S/N of
        it and its two neighbours (root sum of squares) is at least
        ``snr_group``.  Default 2.5.
    snr_group : float, optional
        See ``snr_weak``.  Default 6.
    n_iter : int, optional
        Number of continuum / ripple iterations.  Default 3.

    Returns
    -------
    dict
        ``ripple`` : ndarray, shape (nord, NPIX)
            Fractional ripple ``r``; zero where not corrected.
        ``flat_corrected`` : ndarray, shape (nord, NPIX)
            ``flat / (1 + ripple)``.
        ``normalized`` : ndarray, shape (nord, NPIX)
            Flat divided by its continuum, minus 1.
        ``keep`` : ndarray of bool, shape (nord, NPIX)
            Pixels used in the fits.
        ``sigma`` : ndarray, shape (nord, NPIX)
            Wavenumber in cm^-1 (from the approximate wavelengths).
        ``orders`` : ndarray of int, shape (nord,)
            Echelle order (or row index) of each row.
        ``phase`` : callable
            Carrier phase ``Phi(sigma)`` in cycles.
        ``info`` : dict
            Carrier information (``t_mm``, ``opd_mm_at``, ``fixed``).
        ``report`` : ndarray, shape (nord, 6)
            Per order: order, amplitude [%], S/N, applied (0/1),
            rms before [%], rms after [%].

    Notes
    -----
    Each order is modelled as ``Re[E_i(x) exp(2 pi i Phi(sigma_i(x)))]``:
    the carrier ``Phi`` carries the fast fringe phase, and the low-order
    complex envelope ``E_i`` absorbs flexure shifts, wavelength-solution
    errors, the beat between the two birefringent components and changes of
    fringe contrast.  No flexure correction or exact wavelength solution is
    therefore needed.

    Raises
    ------
    ValueError
        If ``flat`` and ``wave`` do not have the same shape.

    Examples
    --------
    >>> res = deripple(flat, wave, first_order=70)
    >>> clean = res['flat_corrected']   # flat == clean * (1 + res['ripple'])
    """
    flat = np.asarray(flat, float); wave = np.asarray(wave, float)
    if flat.shape != wave.shape:
        raise ValueError(f'flat {flat.shape} and wave {wave.shape} must have the same shape '
                         '(same orders, row for row)')
    nord = flat.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = 1e7 / wave                                   # wavenumber, cm^-1
    orders = (first_order or 0) + np.arange(nord)
    s0 = np.nanmedian(sig)
    good = np.array([usable_mask(r) for r in flat])
    ripple = np.zeros_like(flat)
    for it in range(n_iter):
        norm = np.full_like(flat, np.nan); keep = np.zeros_like(good)
        for i in range(nord):
            if good[i].sum() < 300: continue
            cont, k = fit_continuum(flat[i], good[i], ripple=ripple[i] if it else None)
            norm[i] = flat[i] / cont - 1
            keep[i] = k & np.isfinite(norm[i]) & np.isfinite(sig[i])
        segs = _segments(norm, keep, sig)
        if thickness_mm is None:      # fixed plate, nothing fitted
            Phi, info = fit_carrier_sapphire(segs, (t_range_mm[0] / 10, t_range_mm[1] / 10))
        else:
            t = thickness_mm / 10.
            Phi = lambda s, t=t: 2 * t * n_sapphire(s) * s
            info = dict(t_mm=thickness_mm, opd_mm_at=lambda s, t=t: 2 * t * 10 * ng_sapphire(s), fixed=True)
        ripple = np.zeros_like(flat); fits = {}
        snrs = np.full(nord, np.nan)
        for i in range(nord):
            k = keep[i]
            if k.sum() < 300: continue
            ph = Phi(sig[i])
            lo, hi = X[k].min(), X[k].max()
            xn = np.clip(2 * (X - lo) / (hi - lo) - 1, -1, 1)   # envelope held constant beyond fitted range
            c, amp, w = envelope_fit(xn[k], ph[k], norm[i][k], envelope_degree)
            mod = np.real(L.legval(xn[k], c) * np.exp(2j * np.pi * ph[k]))
            snrs[i] = detection_snr(ph[k], highpass(np.nan_to_num(norm[i]), k)[k], mod)
            fits[i] = (ph, xn, c, amp, w)
        # decision: S/N >= snr_min in the order itself, or a marginal detection
        # (S/N >= snr_weak) supported by the neighbouring orders (the ripple
        # amplitude varies smoothly from order to order)
        s2 = np.nan_to_num(snrs) ** 2
        group = np.sqrt(np.convolve(s2, np.ones(3), 'same'))
        report = []
        for i in range(nord):
            if i not in fits:
                report.append((orders[i], np.nan, np.nan, 0, np.nan, np.nan)); continue
            ph, xn, c, amp, w = fits[i]; k = keep[i]
            applied = snrs[i] >= snr_min or (snrs[i] >= snr_weak and group[i] >= snr_group)
            fin = np.isfinite(flat[i]) & np.isfinite(ph)
            if applied:
                ripple[i][fin] = np.real(L.legval(xn[fin], c) * np.exp(2j * np.pi * ph[fin]))
            rms0 = np.std(norm[i][k][w]); rms1 = np.std((norm[i] - ripple[i])[k][w])
            report.append((orders[i], amp * 100, snrs[i], int(applied), rms0 * 100, rms1 * 100))
        msg = f'iteration {it + 1}: OPD(σ={s0:.0f}) = {info["opd_mm_at"](s0):.4f} mm'
        if 't_mm' in info:
            msg += f'  (sapphire t = {info["t_mm"]:.4f} mm{", fixed" if info.get("fixed") else ""})'
        log.debug(msg)
    report = np.array(report, float)
    log.stdinfo(' order  amp[%]   S/N  applied   rms before -> after [%]')
    for r in report:
        log.stdinfo(f'{int(r[0]):6d}  {r[1]:6.3f} {r[2]:6.1f}    {"yes" if r[3] else " no"}     {r[4]:.3f} -> {r[5]:.3f}')
    return dict(ripple=ripple, flat_corrected=flat / (1 + ripple), normalized=norm, keep=keep,
                sigma=sig, orders=orders, phase=Phi, info=info, report=report)
