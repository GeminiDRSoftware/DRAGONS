#include <stdlib.h>
  double *gspline (double x[], double y[], int n, double u[], int nz)
{
  double spline(), seval();
  double *b,*c,*d,*z;
  double rval;
  int i;

  b = (double *)malloc(sizeof(double[n]));
  c = (double *)malloc(sizeof(double[n]));
  d = (double *)malloc(sizeof(double[n]));

  rval = spline (x,y,b,c,d,n);
  for (i=0;i<nz;i++) {
        z[i] = seval (x,y,b,c,d,u[i],n); 
  }
  free(b);
  free(c);
  free(d);
  return z;

}

  double spline (double x[], double y[], double b[], double c[], double d[], int n)
/*
  Calculate the coefficients b[i], c[i], and d[i], i=0,2,...,n-1
  for a cubic spline interpolation
  s(x) = y[i] + b[i]*(x-x[i]) + c[i]*(x-x[i])**2 + d[i]*(x-x[i])**3
  for  x[i] <= x <= x[i+1]

  Alex Godunov, Last update 16 February 2007


  input ...
    n = the number of data points (n >= 2)
    x = the arrays of data abscissas in strictly increasing order
    y = the arrays of data ordinates

  output ...
    b, c, d  = arrays of spline coefficients.

  comments ...
    the program is based on fortran version of program spline.f
    use function "seval" to evaluate cubic spline interpolation
*/

{
  int    nm1, ib, i;
  double t;

  nm1 = n-2;

  if ( n <= 2 ) return 1.0;
  if ( n <= 3 )
  {
     b[0] = (y[1]-y[0])/(x[1]-x[0]);
     c[0] = 0.0;
     d[0] = 0.0;
     b[1] = b[0];
     c[1] = 0.0;
     d[1] = 0.0;
     return 1.0;
  }
/*
  step 1: preparation
*/
  d[0] = x[1] - x[0];
  c[1] = (y[1] - y[0])/d[0];
  for (i = 1; i <= nm1; i=i+1)
  {
      d[i]   = x[i+1] - x[i];
      b[i]   = 2.0*(d[i-1] + d[i]);
      c[i+1] = (y[i+1] - y[i])/d[i];
      c[i]   = c[i+1] - c[i];
  }
/*
  step 2: end conditions
*/
  b[0]   = -d[0];
  b[n-1] = -d[n-2];
  c[0]   = 0.0;
  c[n-1] = 0.0;
  if ( n != 3 )
  {
       c[0]   = c[2]/(x[3]-x[1]) - c[1]/(x[2]-x[0]);
       c[n-1] = c[n-2]/(x[n-1]-x[n-3]) - c[n-3]/(x[n-2]-x[n-4]);
       c[0]   = c[0]*d[0]*d[0]/(x[3]-x[0]);
       c[n-1] = -c[n-1]*d[n-2]*d[n-2]/(x[n-1]-x[n-4]);
  }
/*
  step 3: forward elimination
*/
  for (i = 1; i <= n-1; i=i+1)
  {
       t = d[i-1]/b[i-1];
       b[i] = b[i] - t*d[i-1];
       c[i] = c[i] - t*c[i-1];
  }
/*
  step 4: back substitution
*/
  c[n-1] = c[n-1]/b[n-1];
  for (ib = 1; ib <= nm1+1; ib=ib+1)
  {
       i = n-ib-1;
       c[i] = (c[i] - d[i]*c[i+1])/b[i];
   }
/*
  step 5: compute polynomial coefficients
*/
  b[n-1] = (y[n-1] - y[nm1])/d[nm1] + d[nm1]*(c[nm1] + 2.*c[n-1]);
  for (i = 0; i <= nm1; i=i+1)
  {
       b[i] = (y[i+1] - y[i])/d[i] - d[i]*(c[i+1] + 2.*c[i]);
       d[i] = (c[i+1] - c[i])/d[i];
       c[i] = 3.*c[i];
  }
  c[n-1] = 3.*c[n-1];
  d[n-1] = d[n-2];
  return 0.0;
}

  double seval(double x[], double y[], double b[], double c[], double d[],
               int n, double u)
/*
  function seval evaluates the cubic spline function
  s(x) = y[i] + b[i]*(u-x[i]) + c[i]*(u-x[i])**2 + d[i]*(u-x[i])**3
  for  x[i] <= u <= x[i+1]

  if  u < x[0]   s = x[0]
  if  u > x[n-1] s = x[n-1]

  input ...
    n = the number of data points
    u = the abscissa at which the spline is to be evaluated
    x,y = the arrays of data abscissas and ordinates
    b,c,d = arrays of spline coefficients computed by spline
*/
{
  int i, j, k;
  double dx, s;
// if u is ouside the x[] interval take a boundary value (left or right)
    if (u <= x[0])   return s = y[0];
    if (u >= x[n-1]) return s = y[n-1];
// a straightforward search to find i so that x[i-1] < x < x[i]
// since the binary (bisectional) search seems to work fine this part is obsolete
/*
    i = 0;
    while (i <= n-1)
    {
     if (x[i] >= u) break;
     i = i + 1;
     }
*/

//  a binary (bisectional) search to find j so that x[j-1] < x < x[j]
//  works much faster and gives same results as a straightforward search

    i = 0;
    j = n;
    while (j > i+1)
    {
       k = (i+j)/2;
       if (u < x[k-1]) j = k;
       else i = k;
    }
//  evaluate spline
  i = i - 1;
  dx = u - x[i];
  s = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]));
  return s;
}

