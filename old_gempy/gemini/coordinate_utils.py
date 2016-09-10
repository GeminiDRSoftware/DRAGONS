import astropy.coordinates
import astropy.units
import astropy._erfa as erfa

def toicrs(frame, ra, dec, equinox=2000.0, ut_datetime=None):
    # Utility function. Converts and RA and Dec in the specified reference frame
    # and equinox at ut_datetime into ICRS. This is used by the ra and dec descriptors.

    # Assume equinox is julian calendar
    equinox = 'J%s' % equinox

    # astropy doesn't understand APPT coordinates. However, it does understand
    # CIRS coordinates, and we can convert from APPT to CIRS by adding the
    # equation of origins to the RA. We can get that using ERFA.
    # To proceed with this, we first let astopy construct the CIRS frame, so
    # that we can extract the obstime object from that to pass to erfa.

    appt_frame = True if frame == 'APPT' else False
    frame = 'cirs' if frame == 'APPT' else frame
    frame = 'fk5' if frame == 'FK5' else frame

    coords = astropy.coordinates.SkyCoord(ra=ra*astropy.units.degree,
                                      dec=dec*astropy.units.degree,
                                      frame=frame,
                                      equinox=equinox,
                                      obstime=ut_datetime)

    if appt_frame:
        # Call ERFA.apci13 to get the Equation of Origin (EO).
        # We just discard the astrom context return
        astrom, eo = erfa.apci13(coords.obstime.jd1, coords.obstime.jd2)
        astrom = None
        # eo comes back as a single element array in radians
        eo = float(eo)
        eo = eo * astropy.units.radian
        # re-create the coords frame object with the corrected ra
        coords = astropy.coordinates.SkyCoord(ra=coords.ra+eo,
                                              dec=coords.dec,
                                              frame=coords.frame.name,
                                              equinox=coords.equinox,
                                              obstime=coords.obstime)

    # Now we can just convert to ICRS...
    icrs = coords.icrs

    # And return values in degrees
    return (icrs.ra.degree, icrs.dec.degree)

