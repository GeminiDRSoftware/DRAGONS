"""
Usage: python mkslitvmod.py <reduced_slitflat>

This script is intended to construct an entirely new SLITVMOD file from
scratch by analysing a reduced slitflat (the slitflat can be reduced
without a slitbias).

It works by identifying the locations of the fibers in the raw
(unrotated) SVC image to calculate the rotation angle, and then transforming
them into locations in the rotated image. It displays both the raw and
rotated SVC images with the fiber locations circled and, if these are
incorrect, it indicates a problem with the calculation.

The output file is called std_slitvmod.fits or high_slitvmod.fits and it
should be renamed slitvmod.fits and placed in a new YYMMDD directory in
geminidr/ghost/lookups/Polyfit/slitv/[std|high] where it will be picked up
automatically when reducing any data with a UT date of YYMMDD or later.

*****
Note that if the change is small, it may not be necessary to run this script
as the new parameter values can be inferred from the existing ones. Notably,
the angle of the pseudoslit must not have changed. In which case one can
measure the change in the location of the slit (determine the center of a
fiber from displaying the old and new SVC images) in each coordinate:
dX and dY where a positive value indicates that the new position is right or
above the old one.

If this change is small then, because the pseudoslit is almost horizontal,
it is possible to simply change the following parameter values:

center_x_blue: add 2*dY
center_x_red:  add 2*dY
center_y_blue: subtract 2*dX
center_y_red:  subtract 2*dX

The reason for the 2 is because the values in the SLITVMOD are in *unbinned*
pixels. And the X/Y swap is because they're in the frame of the *rotated*
SVC image, which is rotated by nearly 90 degrees counter-clockwise.

No changes to other parameters are required, provided the change is small.
Large changes may require the center of rotation (rotxc, rotyc) to be
modified even if the rotation does not change.

NB. HOWEVER THE CHANGES ARE MADE, THEY MUST BE MADE TO BOTH THE STD AND HIGH
RESOLUTION SLIVMOD FILES!
"""

from sys import argv

import astrodata, gemini_instruments

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect
from geminidr.ghost.polyfit import SlitView
from geminidr.ghost.polyfit.slitview import ceildiv

from matplotlib import pyplot as plt

from scipy import ndimage
import numpy as np
from astropy.modeling import models

class SlitView2(SlitView):
    BUFFER = 5

    def __init__(self, slit_image, flat_image, slitvpars, **kwargs):
        regions, nregions = ndimage.label(flat_image > 0.1 * flat_image.max())
        if nregions != 2:
            raise ValueError(f"Found {nregions} illuminated regions "
                             "(expected 2)")

        object_slices = ndimage.find_objects(regions)
        for _slice in object_slices:
            for i, s in enumerate(_slice):
                if (s.start < self.BUFFER + 1 or
                        s.stop >= flat_image.shape[i] - self.BUFFER):
                    raise RuntimeError("Pseudoslit is too close to edge of SVC"
                                       " cutout")

        super().__init__(slit_image, flat_image, slitvpars, **kwargs)

        # Account for the missing ThXe fibre
        if self.mode == 'high':
            object_slices = [(o[0], slice(o[1].start - 6, o[1].stop))
                             for o in object_slices]

        # Blue is brighter than red
        indices = np.argsort([flat_image[_slice].max()
                              for _slice in object_slices])
        self.slit_slices = {}
        self.slit_slices['red'], self.slit_slices['blue'] = [
            object_slices[i] for i in indices]

    def get_raw_slit_position(self, arm):
        """Override base method to return location measured from raw image"""
        _slice = self.slit_slices[arm]
        yc, xc = [0.5 * (s.start + s.stop - 1) for s in _slice]
        return _slice, (yc, xc)


def mkslitvmod(ad_slitflat):
    p = GHOSTSpect([])
    slitv_fn = p._get_slitv_polyfit_filename(ad_slitflat)
    print(slitv_fn)
    slitvpars = astrodata.open(slitv_fn)
    svpars = slitvpars.TABLE
    old_svpars = svpars.copy()
    data = ad_slitflat[0].data
    binning = ad_slitflat.detector_x_bin()
    sview = SlitView2(data, data, slitvpars=svpars[0],
                      mode=ad_slitflat.res_mode(), binning=binning)

    def convert_to_unbinned(coord, integer=None):
        unbinned_coord = coord * binning + 0.5 * (binning - 1)
        if integer is None:
            return unbinned_coord
        # Weirdness: if the coordinate is between pixels, then we want
        # to take the pixel above for a reversed profile, so that the
        # extracted profiles are as similar as possible. We don't worry
        # about that yet
        if integer:
            return (int(unbinned_coord + 0.5 * binning) // binning * binning)
        else:
            return (int(unbinned_coord + 0.5 * binning) // binning * binning)

    slit_model = sview.model_profile(None, data)

    fig, axs = plt.subplots(ncols=2, sharey=True)
    axs[0].imshow(data, cmap='gray_r', aspect='equal', origin='lower')
    centers = {}
    for arm in ('blue', 'red'):
        m = slit_model[arm]
        xc, yc = m.x_center, m.y_center
        sep, angle = m.separation / binning, m.angle
        fibres = np.array(list(m.fibres.values()))
        xc_all = xc - fibres * sep * np.sin(angle * np.pi / 180)
        yc_all = yc + fibres * sep * np.cos(angle * np.pi / 180)
        xc_all += sview.slit_slices[arm][1].start
        yc_all += sview.slit_slices[arm][0].start
        centers[arm] = (xc_all, yc_all)
        for x, y in zip(xc_all, yc_all):
            circle = plt.Circle((x, y), 3, color=arm, fill=False)
            axs[0].add_patch(circle)

    # Check that the rotation angles for the two pseudoslits are similar
    blue_angle, red_angle = [slit_model[arm].angle.value
                             for arm in ('blue', 'red')]
    print("Rotation angles for red, blue (degrees):", blue_angle, red_angle)
    assert abs(blue_angle - red_angle) < 0.1, "Difference in angles is too large"

    # Update parameters and recreate SlitView object
    svpars['rota'] = 0.5 * (blue_angle + red_angle)
    sview = SlitView(data, data, slitvpars=svpars[0],
                     mode=ad_slitflat.res_mode(), binning=binning)
    rot_image = sview.flat_image
    axs[1].imshow(rot_image, cmap='gray_r', aspect='equal', origin='lower')

    y_halflength = int(sview.slit_length / sview.microns_pix / 2)

    rotxc, rotyc = sview.center  # this is how it's done, incorrectly
    for arm in ('blue', 'red'):
        m_rot = models.Rotation2D(angle=-svpars['rota'])
        new_centers = m_rot(centers[arm][0] - rotxc, centers[arm][1] - rotyc)
        xc_all = [convert_to_unbinned(x, integer=False)
                  for x in new_centers[0] + rotxc]
        yc_all = [convert_to_unbinned(y, integer=sview.reverse_profile[arm])
                  for y in new_centers[1] + rotyc]
        assert len(set(xc_all)) == 1, f"Not all x coords the same in {arm}"
        xcent = xc_all[0]
        ycent = np.median(yc_all)
        if sview.mode == 'high':
            ycent += (m.separation + 0.5 * binning) // binning * binning
        svpars[f'center_x_{arm}'] = xcent
        svpars[f'center_y_{arm}'] = ycent
        for x, y in zip(*new_centers):
            circle = plt.Circle((x + rotxc, y + rotyc), 3, color=arm, fill=False)
            axs[1].add_patch(circle)

        # The coordinates along the slit are computed from an upside-down
        # cutout
        for ifu in ('sky', 'obj0', 'obj1'):
            x0 = (xcent - svpars['ext_hw'].data[0]) // binning
            y0 = ycent // binning + y_halflength
            ystart, ystop = [svpars[f'{ifu}pix{i}'].data[0] for i in (0, 1)]
            rwidth = ceildiv(svpars['ext_hw'].data[0], binning) * 2 + 1
            rect = plt.Rectangle((x0 - 0.5, y0 + 0.5 - ystart // binning),
                                 width=rwidth,
                                 height=-(ystop - ystart) // binning,
                                 color=arm, fill=False)
            axs[1].add_patch(rect)

    print("Old and new SLITVMOD parameters:")
    for c in svpars.colnames:
        print(c, old_svpars[c].data, svpars[c].data)

    fig.subplots_adjust(wspace=0)
    plt.tight_layout(pad=0)
    plt.show()

    slitvpars.TABLE = svpars
    slitvpars.write(f"{sview.mode}_slitvmod.fits", overwrite=True)

    # sview = SlitView(data, data, slitvpars=svpars[0],
    #                  mode=ad_slitflat.res_mode(), binning=binning)
    # fig, ax = plt.subplots()
    # for arm in ('blue', 'red'):
    #     profiles = sview.object_slit_profiles(arm=arm, correct_for_sky=False)
    #     for p in profiles:
    #         ax.plot(p, color=arm)
    # plt.show()


if __name__ == "__main__":
    try:
        filename = argv[1]
    except IndexError:
        raise("Must provide a filename of a reduced slitflat")
    slitflat = astrodata.open(filename)
    assert {'FLAT', 'PROCESSED', 'SLIT'}.issubset(slitflat.tags), \
        f"{slitflat.filename} is not a SLITFLAT"
    mkslitvmod(slitflat)
