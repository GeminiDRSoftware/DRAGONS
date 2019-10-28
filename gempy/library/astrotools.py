"""
The astroTools module contains astronomy specific utility functions
"""
from __future__ import print_function

import os
import re
import numpy as np
from collections import namedtuple
from astropy import units as u

Section = namedtuple('Section', 'x1 x2 y1 y2')

def array_from_list(list_of_quantities):
    """
    Convert a list of Quantity objects to a numpy array. The elements of the
    input list must all be converted to the same units.

    Parameters
    ----------
    list_of_quantities: list
        Quantities objects that all have equivalencies

    Returns
    -------
    array: array representation of this list
    """
    unit = list_of_quantities[0].unit
    values = [x.to(unit).value for x in list_of_quantities]
    # subok=True is needed to handle magnitude/log units
    return u.Quantity(np.array(values), unit, subok=True)

def boxcar(data, operation=np.median, size=1):
    """
    "Smooth" a 1D array by applying a boxcar filter along it. Any operation
    can be performed, as long as it can take a sequence and return a single
    value.

    Parameters
    ----------
    data: 1D ndarray
        the data to be maninpulated
    operation: callable
        function for the boxcar to use
    size: int
        the boxcar width will be 2*size+1

    Returns
    -------
    1D ndarray: same shape as data, after the boxcar operation
    """
    try:
        boxarray = np.array([operation(data[max(i-size, 0):i+size+1])
                             for i in range(len(data))])
    except ValueError:  # Handle things like np.logical_and
        boxarray = np.array([operation.reduce(data[max(i-size, 0):i+size+1])
                             for i in range(len(data))])
    return boxarray

def divide0(numerator, denominator):
    """
    Perform division, replacing division by zero with zero. This expands
    on the np.divide() function by having to deal with cases where either
    the numerator and/or denominator might be scalars, rather than arrays,
    and also deals with cases where they might be integer types.

    Parameters
    ----------
    numerator: float/array-like
        the numerator for the division
    denominator: float/array-like
        the denominator for the division

    Returns
    -------
    The quotient, with instances where the denominator is zero replace by zero
    """
    try:
        is_int = np.issubdtype(denominator.dtype, np.integer)
    except AttributeError:
        # denominator is a scalar
        try:
            is_int = np.issubdtype(numerator.dtype, np.integer)
        except AttributeError:
            # numerator is also a scalar
            return 0 if denominator == 0 else numerator / denominator
        else:
            dtype = np.float32 if is_int else numerator.dtype
            return np.divide(numerator, denominator, out=np.zeros(numerator.shape, dtype=dtype), where=denominator!=0)
    else:
        dtype = np.float32 if is_int else denominator.dtype
        return np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=dtype), where=denominator!=0)

def rasextodec(string):
    """
    Convert hh:mm:ss.sss to decimal degrees
    """
    match_ra = re.match(r"(\d+):(\d+):(\d+\.\d+)", string)
    if match_ra:
        hours = float(match_ra.group(1))
        minutes = float(match_ra.group(2))
        secs = float(match_ra.group(3))

        minutes += (secs/60.0)
        hours += (minutes/60.0)

        degrees = hours * 15.0
    else:
        raise ValueError('Invalid RA string')

    return degrees

def degsextodec(string):
    """
    Convert [-]dd:mm:ss.sss to decimal degrees
    """
    match_dec = re.match(r"(-*)(\d+):(\d+):(\d+\.\d+)", string)
    if match_dec:
        sign = match_dec.group(1)
        if sign == '-':
            sign = -1.0
        else:
            sign = +1.0

        degs = float(match_dec.group(2))
        minutes = float(match_dec.group(3))
        secs = float(match_dec.group(4))

        minutes += (secs/60.0)
        degs += (minutes/60.0)

        degs *= sign
    else:
        raise ValueError('Invalid Dec string')

    return degs

def section_str_to_tuple(section, log=None):
    warn = log.warning if log else print
    if section is not None:
        try:
            x1, x2, y1, y2 = [int(v) for v in section.strip('[]').
                replace(',', ':').split(':')]
        except (AttributeError, ValueError):
            warn("Cannot parse section. Using full frame for statistics")
            section = None
        else:
            section = Section(x1-1, x2, y1-1, y2)
    return section

def get_corners(shape):
    """
    This is a recursive function to calculate the corner indices
    of an array of the specified shape.

    :param shape: length of the dimensions of the array
    :type shape: tuple of ints, one for each dimension

    """
    if not type(shape) == tuple:
        raise TypeError('get_corners argument is non-tuple')

    if len(shape) == 1:
        corners = [(0,), (shape[0]-1,)]
    else:
        shape_less1 = shape[1:len(shape)]
        corners_less1 = get_corners(shape_less1)
        corners = []
        for corner in corners_less1:
            newcorner = (0,) + corner
            corners.append(newcorner)
            newcorner = (shape[0]-1,) + corner
            corners.append(newcorner)

    return corners

def rotate_2d(degs):
    """
    Little helper function to return a basic 2-D rotation matrix.

    :param degs: rotation amount, in degrees
    :type degs: float
    """
    rads = np.radians(degs)
    sine = np.sin(rads)
    cosine = np.cos(rads)
    return np.array([[cosine, -sine], [sine, cosine]])


def clipped_mean(data):
    num_total = len(data)
    mean = data.mean()
    sigma = data.std()

    if num_total < 3:
        return mean, sigma

    num = num_total
    clipped_data = data
    clip = 0
    while num > 0.5 * num_total:
        # CJS: edited this as upper limit was mean+1*sigma => bias
        clipped_data = data[(data < mean + 3*sigma) & (data > mean - 3*sigma)]
        num = len(clipped_data)

        if num > 0:
            mean = clipped_data.mean()
            sigma = clipped_data.std()
        elif clip == 0:
            return mean, sigma
        else:
            break

        clip += 1
        if clip > 10:
            break

    return mean, sigma


# The following functions and classes were borrowed from STSCI's spectools
# package, currently under development.  They might be able to be
# replaced with a direct import of spectools.util if/when it is available

IRAF_MODELS_MAP = {1.: 'chebyshev',
                   2.: 'legendre',
                   3.: 'spline3',
                   4.: 'spline1'}
INVERSE_IRAF_MODELS_MAP = {'chebyshev': 1.,
                           'legendre': 2.,
                           'spline3': 3.,
                           'spline1': 4.}

def get_records(fname):
    """
    Read the records of an IRAF database file ionto a python list

    Parameters
    ----------
    fname: string
           name of an IRAF database file

    Returns
    -------
        A list of records
    """
    filehandle = open(fname)
    dtb = filehandle.read()
    filehandle.close()
    records = []
    recs = dtb.split('begin')[1:]
    records = [Record(r) for r in recs]
    return records

def get_database_string(fname):
    """
    Read an IRAF database file

    Parameters
    ----------
    fname: string
          name of an IRAF database file

    Returns
    -------
        the database file as a string
    """
    f = open(fname)
    dtb = f.read()
    f.close()
    return dtb

class Record(object):
    """
    A base class for all records - represents an IRAF database record

    Attributes
    ----------
    recstr: string
            the record as a string
    fields: dict
            the fields in the record
    taskname: string
            the name of the task which created the database file
    """
    def __init__(self, recstr):
        self.recstr = recstr
        self.fields = self.get_fields()
        self.taskname = self.get_task_name()

    def aslist(self):
        reclist = self.recstr.split('\n')
        reclist = [l.strip() for l in reclist]
        out = [reclist.remove(l) for l in reclist if len(l) == 0]
        return reclist

    def get_fields(self):
        # read record fields as an array
        fields = {}
        flist = self.aslist()
        numfields = len(flist)
        for i in range(numfields):
            line = flist[i]
            if line and line[0].isalpha():
                field = line.split()
                if i+1 < numfields:
                    if not flist[i+1][0].isalpha():
                        fields[field[0]] = self.read_array_field(
                                             flist[i:i+int(field[1])+1])
                    else:
                        fields[field[0]] = " ".join(s for s in field[1:])
                else:
                    fields[field[0]] = " ".join(s for s in field[1:])
            else:
                continue
        return fields

    def get_task_name(self):
        try:
            return self.fields['task']
        except KeyError:
            return None

    def read_array_field(self, fieldlist):
        # Turn an iraf record array field into a numpy array
        fieldline = [l.split() for l in fieldlist[1:]]
        # take only the first 3 columns
        # identify writes also strings at the end of some field lines
        xyz = [l[:3] for l in fieldline]
        try:
            farr = np.array(xyz)
        except:
            print("Could not read array field %s" % fieldlist[0].split()[0])
        return farr.astype(np.float64)

class IdentifyRecord(Record):
    """
    Represents a database record for the longslit.identify task

    Attributes
    ----------
    x: array
       the X values of the identified features
       this represents values on axis1 (image rows)
    y: int
       the Y values of the identified features
       (image columns)
    z: array
       the values which X maps into
    modelname: string
        the function used to fit the data
    nterms: int
        degree of the polynomial which was fit to the data
        in IRAF this is the number of coefficients, not the order
    mrange: list
        the range of the data
    coeff: array
        function (modelname) coefficients
    """
    def __init__(self, recstr):
        super(IdentifyRecord, self).__init__(recstr)
        self._flatcoeff = self.fields['coefficients'].flatten()
        self.x = self.fields['features'][:, 0]
        self.y = self.get_ydata()
        self.z = self.fields['features'][:, 1]
####here - ref?
        self.zref = self.fields['features'][:, 2]
        self.modelname = self.get_model_name()
        self.nterms = self.get_nterms()
        self.mrange = self.get_range()
        self.coeff = self.get_coeff()

    def get_model_name(self):
        return IRAF_MODELS_MAP[self._flatcoeff[0]]

    def get_nterms(self):
        return self._flatcoeff[1]

    def get_range(self):
        low = self._flatcoeff[2]
        high = self._flatcoeff[3]
        return [low, high]

    def get_coeff(self):
        return self._flatcoeff[4:]

    def get_ydata(self):
        image = self.fields['image']
        left = image.find('[')+1
        right = image.find(']')
        section = image[left:right]
        if ',' in section:
            yind = image.find(',')+1
            return int(image[yind:-1])
        else:
            return int(section)
        #xind = image.find('[')+1
        #yind = image.find(',')+1
        #return int(image[yind:-1])

class FitcoordsRecord(Record):
    """
    Represents a database record for the longslit.fitccords task

    Attributes
    ----------
    modelname: string
        the function used to fit the data
    xorder: int
        number of terms in x
    yorder: int
        number of terms in y
    xbounds: list
        data range in x
    ybounds: list
        data range in y
    coeff: array
        function coefficients

    """
    def __init__(self, recstr):
        super(FitcoordsRecord, self).__init__(recstr)
        self._surface = self.fields['surface'].flatten()
        self.modelname = IRAF_MODELS_MAP[self._surface[0]]
        self.xorder = self._surface[1]
        self.yorder = self._surface[2]
        self.xbounds = [self._surface[4], self._surface[5]]
        self.ybounds = [self._surface[6], self._surface[7]]
        self.coeff = self.get_coeff()

    def get_coeff(self):
        return self._surface[8:]

class IDB(object):
    """
    Base class for an IRAF identify database

    Attributes
    ----------
    records: list
             a list of all `IdentifyRecord` in the database
    numrecords: int
             number of records
    """
    def __init__(self, dtbstr):
        lst = self.aslist(dtbstr)
        self.records = [IdentifyRecord(rstr) for rstr in self.aslist(dtbstr)]
        self.numrecords = len(self.records)

    def aslist(self, dtb):
        # return a list of records
        # if the first one is a comment remove it from the list
        record_list = dtb.split('begin')
        try:
            rl0 = record_list[0].split('\n')
        except:
            return record_list
        if len(rl0) == 2 and rl0[0].startswith('#') and not rl0[1].strip():
            return record_list[1:]
        elif len(rl0) == 1 and not rl0[0].strip():
            return record_list[1:]
        else:
            return record_list

class ReidentifyRecord(IDB):
    """
    Represents a database record for the onedspec.reidentify task
    """
    def __init__(self, databasestr):
        super(ReidentifyRecord, self).__init__(databasestr)
        self.x = np.array([r.x for r in self.records])
        self.y = self.get_ydata()
        self.z = np.array([r.z for r in self.records])


    def get_ydata(self):
        y = np.ones(self.x.shape)
        y = y * np.array([r.y for r in self.records])[:, np.newaxis]
        return y


# This class pulls together fitcoords and identify databases into
# a single entity that can be written to or read from disk files
# or pyfits binary tables
class SpectralDatabase(object):
    def __init__(self, database_name=None, record_name=None,
                 binary_table=None):
        """
        database_name is the name of the database directory
        on disk that contains the database files associated with
        record_name.  For example, database_name="database",
        record_name="image_001" (corresponding to the first science
        extention in a data file called image.fits
        """
        self.database_name = database_name
        self.record_name = record_name
        self.binary_table = binary_table

        self.identify_database = None
        self.fitcoords_database = None

        # Initialize from database on disk
        if database_name is not None and record_name is not None:

            if not os.path.isdir(database_name):
                raise IOError('Database directory %s does not exist' %
                              database_name)

            # Read in identify database
            db_filename = "%s/id%s" % (database_name, record_name)
            if not os.access(db_filename, os.R_OK):
                raise IOError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.identify_database = IDB(db_str)

            # Read in fitcoords database
            db_filename = "%s/fc%s" % (database_name, record_name)
            if not os.access(db_filename, os.R_OK):
                raise IOError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.fitcoords_database = FitcoordsRecord(db_str)

        # Initialize from pyfits binary table in memory
        elif binary_table is not None:

            # Get record_name from header if not passed
            if record_name is not None:
                self.record_name = record_name
            else:
                self.record_name = binary_table.header["RECORDNM"]

            # Format identify information from header and table
            # data into a database string
            db_str = self._identify_db_from_table(binary_table)
            self.identify_database = IDB(db_str)

            # Format fitcoords information from header
            # into a database string
            db_str = self._fitcoords_db_from_table(binary_table)
            self.fitcoords_database = FitcoordsRecord(db_str)

        else:
            raise TypeError("Both database and binary table are None.")

    def _identify_db_from_table(self, tab):

        # Get feature information from table data
        features = tab.data
        nrows = len(features)
        nfeat = features["spectral_coord"].shape[1]
        ncoeff = features["fit_coefficients"].shape[1]

        db_str = ""

        for row in range(nrows):

            feature = features[row]

            # Make a dictionary to hold information gathered from
            # the table.  This structure is not quite the same as
            # the fields member of the Record class, but it is the
            # same principle
            fields = {}
            fields["id"] = self.record_name
            fields["task"] = "identify"
            fields["image"] = "%s[*,%d]" % (self.record_name,
                                            feature["spatial_coord"])
            fields["units"] = tab.header["IDUNITS"]

            zip_feature = np.array([feature["spectral_coord"],
                                    feature["fit_wavelength"],
                                    feature["ref_wavelength"]])
            fields["features"] = zip_feature.swapaxes(0, 1)

            fields["function"] = tab.header["IDFUNCTN"]
            fields["order"] = tab.header["IDORDER"]
            fields["sample"] = tab.header["IDSAMPLE"]
            fields["naverage"] = tab.header["IDNAVER"]
            fields["niterate"] = tab.header["IDNITER"]

            reject = tab.header["IDREJECT"].split()
            fields["low_reject"] = float(reject[0])
            fields["high_reject"] = float(reject[1])
            fields["grow"] = tab.header["IDGROW"]

            # coefficients is a list of numbers with the following elements:
            # 0: model number (function type)
            # 1: order
            # 2: x min
            # 3: x max
            # 4 on: function coefficients
            coefficients = []

            model_num = INVERSE_IRAF_MODELS_MAP[fields["function"]]
            coefficients.append(model_num)

            coefficients.append(fields["order"])

            idrange = tab.header["IDRANGE"].split()
            coefficients.append(float(idrange[0]))
            coefficients.append(float(idrange[1]))

            fit_coeff = feature["fit_coefficients"].tolist()
            coefficients.extend(fit_coeff)
            fields["coefficients"] = np.array(coefficients).astype(np.float64)


            # Compose fields into a single string
            rec_str = "%-8s%-8s %s\n" % \
                      ("begin", fields["task"], fields["image"])
            for field in ["id", "task", "image", "units"]:
                rec_str += "%-8s%-8s%s\n" % ("", field, str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % \
                       ("", "features", len(fields["features"]))
            for feat in fields["features"]:
                rec_str += "%16s%10f %10f %10f\n" % \
                           ("", feat[0], feat[1], feat[2])
            for field in ["function", "order", "sample",
                          "naverage", "niterate", "low_reject",
                          "high_reject", "grow"]:
                rec_str += "%-8s%s %s\n" % ("", field, str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % ("", "coefficients",
                                         len(fields["coefficients"]))
            for coeff in fields["coefficients"]:
                rec_str += "%-8s%-8s%E\n" % ("", "", coeff)
            rec_str += "\n"

            db_str += rec_str

        return db_str

    def _fitcoords_db_from_table(self, tab):

        # Make a dictionary to hold information gathered from
        # the table.  This structure is not quite the same as
        # the fields member of the Record class, but it is the
        # same principle
        fields = {}

        fields["begin"] = self.record_name
        fields["task"] = "fitcoords"
        fields["axis"] = tab.header["FCAXIS"]
        fields["units"] = tab.header["FCUNITS"]

        # The surface is a list of numbers with the following elements:
        # 0: model number (function type)
        # 1: x order
        # 2: y order
        # 3: cross-term type (always 1. for fitcoords)
        # 4. xmin
        # 5: xmax
        # 6. xmin
        # 7: xmax
        # 8 on: function coefficients
        surface = []

        model_num = INVERSE_IRAF_MODELS_MAP[tab.header["FCFUNCTN"]]
        surface.append(model_num)

        xorder = tab.header["FCXORDER"]
        yorder = tab.header["FCYORDER"]
        surface.append(xorder)
        surface.append(yorder)
        surface.append(1.)

        fcxrange = tab.header["FCXRANGE"].split()
        surface.append(float(fcxrange[0]))
        surface.append(float(fcxrange[1]))
        fcyrange = tab.header["FCYRANGE"].split()
        surface.append(float(fcyrange[0]))
        surface.append(float(fcyrange[1]))

        for i in range(int(xorder)*int(yorder)):
            coeff = tab.header["FCCOEF%d" % i]
            surface.append(coeff)

        fields["surface"] = np.array(surface).astype(np.float64)

        # Compose fields into a single string
        db_str = "%-8s%s\n" % ("begin", fields["begin"])
        for field in ["task", "axis", "units"]:
            db_str += "%-8s%-8s%s\n" % ("", field, str(fields[field]))
        db_str += "%-8s%-8s%d\n" % ("", "surface", len(fields["surface"]))
        for coeff in fields["surface"]:
            db_str += "%-8s%-8s%E\n" % ("", "", coeff)

        return db_str

    def write_to_disk(self, database_name=None, record_name=None):

        # Check for provided names; use names from self if not
        # provided as input
        if database_name is None and self.database_name is None:
            raise TypeError("No database_name provided")
        elif database_name is None and self.database_name is not None:
            database_name = self.database_name
        if record_name is None and self.record_name is None:
            raise TypeError("No record_name provided")
        elif record_name is None and self.record_name is not None:
            record_name = self.record_name

        # Make the directory if needed
        if not os.path.exists(database_name):
            os.mkdir(database_name)

        # Timestamp
        import datetime
        timestamp = str(datetime.datetime.now())

        # Write identify files
        id_db = self.identify_database
        if id_db is not None:
            db_filename = "%s/id%s" % (database_name, record_name)
            db_file = open(db_filename, "w")
            db_file.write("# "+timestamp+"\n")
            for record in id_db.records:
                db_file.write("begin")
                db_file.write(record.recstr)
            db_file.close()

        # Write fitcoords files
        fc_db = self.fitcoords_database
        if fc_db is not None:
            db_filename = "%s/fc%s" % (database_name, record_name)
            db_file = open(db_filename, "w")
            db_file.write("# "+timestamp+"\n")
            db_file.write(fc_db.recstr)
            db_file.close()

    def as_binary_table(self, record_name=None):

        # Should this be lazy loaded?
        import astropy.io.fits as pf

        if record_name is None:
            record_name = self.record_name

        # Get the maximum number of features identified in any
        # record.  Use this as the length of the array in the
        # wavelength_coord and fit_wavelength fields
        nfeat = max([len(record.x)
                     for record in self.identify_database.records])

        # The number of coefficients should be the same for all
        # records, so take the value from the first record
        ncoeff = self.identify_database.records[0].nterms

        # Get the number of rows from the number of identify records
        nrows = self.identify_database.numrecords

        # Create pyfits Columns for the table
        column_formats = [{"name":"spatial_coord", "format":"I"},
                          {"name":"spectral_coord", "format":"%dE"%nfeat},
                          {"name":"fit_wavelength", "format":"%dE"%nfeat},
                          {"name":"ref_wavelength", "format":"%dE"%nfeat},
                          {"name":"fit_coefficients", "format":"%dE"%ncoeff},]
        columns = [pf.Column(**fmt) for fmt in column_formats]

        # Make the empty table.  Use the number of records in the
        # database as the number of rows
        table = pf.new_table(columns, nrows=nrows)

        # Populate the table from the records
        for i in range(nrows):
            record = self.identify_database.records[i]
            row = table.data[i]
            row["spatial_coord"] = record.y
            row["fit_coefficients"] = record.coeff
            if len(row["spectral_coord"]) != len(record.x):
                row["spectral_coord"][:len(record.x)] = record.x
                row["spectral_coord"][len(record.x):] = -999
            else:
                row["spectral_coord"] = record.x
            if len(row["fit_wavelength"]) != len(record.z):
                row["fit_wavelength"][:len(record.z)] = record.z
                row["fit_wavelength"][len(record.z):] = -999
            else:
                row["fit_wavelength"] = record.z
            if len(row["ref_wavelength"]) != len(record.zref):
                row["ref_wavelength"][:len(record.zref)] = record.zref
                row["ref_wavelength"][len(record.zref):] = -999
            else:
                row["ref_wavelength"] = record.zref

        # Store the record name in the header
        table.header.update("RECORDNM", record_name)

        # Store other important values from the identify records in the header
        # These should be the same for all records, so take values
        # from the first record
        first_record = self.identify_database.records[0]
        table.header.update("IDUNITS", first_record.fields["units"])
        table.header.update("IDFUNCTN", first_record.modelname)
        table.header.update("IDORDER", first_record.nterms)
        table.header.update("IDSAMPLE", first_record.fields["sample"])
        table.header.update("IDNAVER", first_record.fields["naverage"])
        table.header.update("IDNITER", first_record.fields["niterate"])
        table.header.update("IDREJECT", "%s %s" %
                            (first_record.fields["low_reject"],
                             first_record.fields["high_reject"]))
        table.header.update("IDGROW", first_record.fields["grow"])
        table.header.update("IDRANGE", "%s %s" %
                            (first_record.mrange[0], first_record.mrange[1]))

        # Store fitcoords information in the header
        fc_record = self.fitcoords_database
        table.header.update("FCUNITS", fc_record.fields["units"])
        table.header.update("FCAXIS", fc_record.fields["axis"])
        table.header.update("FCFUNCTN", fc_record.modelname)
        table.header.update("FCXORDER", fc_record.xorder)
        table.header.update("FCYORDER", fc_record.yorder)
        table.header.update("FCXRANGE", "%s %s" %
                            (fc_record.xbounds[0], fc_record.xbounds[1]))
        table.header.update("FCYRANGE", "%s %s" %
                            (fc_record.ybounds[0], fc_record.ybounds[1]))
        for i in range(len(fc_record.coeff)):
            coeff = fc_record.coeff[i]
            table.header.update("FCCOEF%d" % i, coeff)
####here -- comments

        return table
