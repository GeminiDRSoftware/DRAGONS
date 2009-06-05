import numpy as np

 # $Id: c_correlate.pro,v 1.20 2004/01/21 15:54:48 scottm Exp $
#
# Copyright (c) 1995-2004, Research Systems, Inc.  All rights reserved.
#       Unauthorized reproduction prohibited.
#+
# NAME:
#       C_CORRELATE
#
# PURPOSE:
#       This function computes the cross correlation Pxy(L) or cross
#       covariance Rxy(L) of two sample populations X and Y as a function
#       of the lag (L).
#
# CATEGORY:
#       Statistics.
#
# CALLING SEQUENCE:
#       Result = C_correlate(X, Y, Lag)
#
# INPUTS:
#       X:    An n-element vector of type integer, float or double.
#
#       Y:    An n-element vector of type integer, float or double.
#
#     LAG:    A scalar or n-element vector, in the interval [-(n-2), (n-2)],
#             of type integer that specifies the absolute distance(s) between
#             indexed elements of X.
#
# KEYWORD PARAMETERS:
#       COVARIANCE:    If set to a non-zero value, the sample cross
#                      covariance is computed.
#
#       DOUBLE:        If set to a non-zero value, computations are done in
#                      double precision arithmetic.
#
# EXAMPLE
#       Define two n-element sample populations.
#         x = [3.73, 3.67, 3.77, 3.83, 4.67, 5.87, 6.70, 6.97, 6.40, 5.57]
#         y = [2.31, 2.76, 3.02, 3.13, 3.72, 3.88, 3.97, 4.39, 4.34, 3.95]
#
#       Compute the cross correlation of X and Y for LAG = -5, 0, 1, 5, 6, 7
#         lag = [-5, 0, 1, 5, 6, 7]
#         result = c_correlate(x, y, lag)
#
#       The result should be:
#         [-0.428246, 0.914755, 0.674547, -0.405140, -0.403100, -0.339685]
#
# PROCEDURE:
#       See computational formula published in IDL manual.
#
# REFERENCE:
#       INTRODUCTION TO STATISTICAL TIME SERIES
#       Wayne A. Fuller
#       ISBN 0-471-28715-6
#
# MODIFICATION HISTORY:
#       Written by:  GGS, RSI, October 1994
#       Modified:    GGS, RSI, August 1995
#                    Corrected a condition which excluded the last term of the
#                    time-series.
#                - GGS, RSI, April 1996
#                    Simplified CROSS_COV function. Added DOUBLE keyword.
#                    Modified keyword checking and use of double precision.
#                - W. Biagiotti,  Advanced Testing Technologies
#                Inc., Hauppauge, NY, July 1997, Moved all
#                constant calculations out of main loop for
#                greatly reduced processing time.
#   CT, RSI, September 2002. Further speed improvements, per W. Biagiotti.
#                Now handles large vectors and complex inputs.
#-
def c_correlate(x, y, lag, covariance=None, double=None):

   n_params = 3
   doublein = double
   
   # COMPILE_OPT IDL2
   
   # Compute the sample cross correlation or cross covariance of
   # (Xt, Xt+l) and (Yt, Yt+l) as a function of the lag (l).
   
   # ON_ERROR, 2
   
   x = np.asfarray(x)
   y = np.asfarray(y)
   nx = np.size(x)
   
   if (nx != np.size(y)):
      print("X and Y arrays must have the same number of elements.")
   
   #Check length.
   if (nx < 2):   
       print("X and Y arrays must contain 2 or more elements.")
   
   
       #If the DOUBLE keyword is not set then the internal precision and
       #result are identical to the type of input.
   usedouble = doublein is not None
   tylag = type(lag)
   if usedouble:
     tylag = np.double 
   
   # This will now be in double precision if Double is set.
   xd = x - np.sum(x, dtype=tylag) / nx #Deviations
   yd = y - np.sum(y, dtype=tylag) / nx
   
   nlag = np.size(lag)

   cross = np.zeros(nlag, dtype=tylag)
   
   m = np.absolute(lag)
   for k in range(nlag):
       # Note the reversal of the variables for negative lags.
       cross[k] = (((lag[k] >= 0)) and [np.sum(xd[0:(nx - lag[k] - 1)+1] * yd[lag[k]:])] or [np.sum(yd[0:(nx + lag[k] - 1)+1] * xd[-lag[k]:])])[0]
   
   # Divide by N for covariance, or divide by variance for correlation.
   temp = np.asfarray(cross.copy())
   if covariance is not None:
       cross = temp / nx
   else:
       cross = temp / np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2))
   del(temp)
   
   return ((usedouble) and [cross] or [np.asfarray(cross)])[0]
   


