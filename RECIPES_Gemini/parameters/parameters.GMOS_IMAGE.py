#
from astrodata.ParamObject import PrimitiveParameter

localParameterIndex = {
                       
                      "prepare":{'theta':PrimitiveParameter('theta', True, helps="For testing."), 
                                 'sigma':PrimitiveParameter('sigma', False, helps="For testing.")},
                      "biasSub":{'openthe':PrimitiveParameter('openthe', True, helps="For testing"),
                                 'podbaydoor':PrimitiveParameter('podbaydoor', True, helps="For testing"),
                                 'hal':PrimitiveParameter('hal', True, helps="For testing")}
                      
                      }
