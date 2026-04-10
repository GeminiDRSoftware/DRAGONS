Parameter defaults and options
------------------------------
::

   suffix               '_stack'             Filename suffix
   apply_dq             True                 Use DQ to mask bad pixels?
   statsec              None                 Section for statistics (1-indexed, inclusive-max, x-first)
   operation            'mean'               Averaging operation
      Allowed values:
      	mean	arithmetic mean
      	wtmean	variance-weighted mean
      	median	median
      	lmedian	low-median
      
   reject_method        'varclip'            Pixel rejection method
      Allowed values:
      	none	no rejection
      	minmax	reject highest and lowest pixels
      	sigclip	reject pixels based on scatter
      	varclip	reject pixels based on variance array
      
   hsigma               3.0                  High rejection threshold (sigma)
      	Valid Range = [0,inf)
   lsigma               3.0                  Low rejection threshold (sigma)
      	Valid Range = [0,inf)
   mclip                True                 Use median for sigma-clipping?
   max_iters            None                 Maximum number of clipping iterations
      	Valid Range = [1,inf)
   nlow                 0                    Number of low pixels to reject
      	Valid Range = [0,inf)
   nhigh                0                    Number of high pixels to reject
      	Valid Range = [0,inf)
   memory               1.0                  Memory available for stacking (GB)
      	Valid Range = [0.01,inf)
   save_rejection_map   False                Save rejection map?
   separate_ext         True                 Handle extensions separately?
