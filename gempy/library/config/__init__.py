#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from .config import *
from .rangeField import *
from .choiceField import *
from .listField import *
from .dictField import *
from .configField import *
from .configChoiceField import *
from .configurableField import *
from .configDictField import *
#from .convert import *
from .wrap import *
from .registry import *
#from .version import *


class core_1Dfitting_config(Config):
    function = ChoiceField("Fitting function", str,
                           allowed={"spline3": "Cubic spline",
                                    "chebyshev": "Chebyshev polynomial"},
                           default="spline3", optional=False)
    order = RangeField("Order of fitting function", int, 6, min=1)
    lsigma = RangeField("Low rejection in sigma of fit", float, 3,
                        min=0, optional=True)
    hsigma = RangeField("High rejection in sigma of fit", float, 3,
                        min=0, optional=True)
    niter = RangeField("Maximum number of rejection iterations", int, None,
                       min=0, optional=True)
    grow = RangeField("Rejection growing radius", float, 0, min=0)
