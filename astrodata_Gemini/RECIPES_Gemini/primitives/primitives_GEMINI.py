from primitives_GENERAL import GENERALPrimitives
from primitives_bookkeeping import BookkeepingPrimitives
from primitives_display import DisplayPrimitives
from primitives_photometry import PhotometryPrimitives
from primitives_preprocessing import PreprocessingPrimitives
from primitives_qa import QAPrimitives
from primitives_registration import RegistrationPrimitives
from primitives_resample import ResamplePrimitives
from primitives_stack import StackPrimitives
from primitives_standardization import StandardizationPrimitives
import datetime
import time

class GEMINIPrimitives(BookkeepingPrimitives,DisplayPrimitives,
                       PhotometryPrimitives,PreprocessingPrimitives,
                       QAPrimitives,RegistrationPrimitives,
                       ResamplePrimitives,StackPrimitives,
                       StandardizationPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    datacounter = 0
    def testReportQAMetric(self, rc):
        import random
        import time
        bigreport = {"msgtype": "qametric",
         "timestamp": time.time(),
         "iq": {"band": 85,
                "delivered": 0.983,
                "delivered_error": 0.1,
                "zenith": 0.7 + random.uniform(-.5,.5),#0.947,
                "ellipticity": 0.118,
                "ellip_error": 0.067,
                "requested": 85,
                "comment": ["High ellipticity"],
                },
         "cc": {"band": 70,
                "zeropoint": {"e2v 10031-23-05, right":{"value":26.80,
                                                        "error":0.05},
                              "e2v 10031-01-03, right":{"value":26.86,
                                                        "error":0.03},
                              "e2v 10031-01-03, left":{"value":26.88,
                                                       "error":0.06}},
                "extinction": .5 + random.uniform(-.5,.5),#0.02,
                "extinction_error": 0.5,
                "requested": 50,
                "comment": ["Requested CC not met"],
                },
         "bg": {"band": 100,
                "brightness": 20 + random.uniform(-.8,.8),#19.17,
                "brightness_error": 0.5,
                "requested": 100,
                "comment": []
                },
         }
        from copy import copy
        import pprint
        def mock_metadata(ad, numcall = 0):
            mtd = {"metadata":
                    { "filename": ad.filename,
                      "datalabel": ad.data_label().as_pytype(),
                      "local_time": ad.local_time().as_pytype(),
                      "ut_time": ad.ut_datetime().as_pytype().strftime("%Y-%m-%d %H:%M:%S.%f"),
                      "wavelength": ad.central_wavelength(asMicrometers=True).as_pytype(),
                      "filter": ad.filter_name(pretty=True).as_pytype(),
                      "waveband": ad.wavelength_band().as_pytype(),
                      "airmass": ad.airmass().as_pytype(),
                      "instrument": ad.instrument().as_pytype(),
                      "object": ad.object().as_pytype(),
                      "types": ad.get_types(),
                    }
                  }
            import random
            now = datetime.datetime.utcnow()
            if now.hour>17:
                tonight = now.replace(hour =5, minute=0)+datetime.timedelta(days=1)
            else:
                tonight = now.replace(hour =5, minute=0)
            nexttime = datetime.timedelta(minutes = self.datacounter*15 + random.randint(-60,0))
            self.datacounter += 1
            now_ut = tonight + nexttime

            filename = "N%sS0%0.3d.fits" % (now.strftime("%Y%m%d"),
                                            random.randint(1,999))
            dl = "GN-2012B-Q-%i-1-001" % random.randint(1,20)

            wlen = ['g','V','r','R','i','I','z','I']
            #wlen = ['u','U','b','B','g','V','r','R','i','I','z','I','Y','Y']
            #wlen = ['g','V']
            wlen_ind = 2*random.randint(0,len(wlen)/2-1)
            
            mtd["metadata"].update({"ut_time": now_ut.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                    "datalabel": dl,
                                    "filename": filename,
                                    "filter": wlen[wlen_ind],
                                    "wavelength": wlen[wlen_ind],
                                    "waveband": wlen[wlen_ind+1]})
            delt = (now_ut - now.replace(hour=0,minute=0) )
            nowsec = float(delt.days*86400 + delt.seconds)
                
            return (mtd, nowsec)
        from time import sleep
        from math import sin
        if "test_num" in rc:
                test_num = int(rc["test_num"])
        else:
                test_num = 1
        if "test_burst" in rc:
            test_burst = int(rc["test_burst"])
        else:
                test_burst = 1
        if "test_sleep" in rc:
            test_sleep = float(rc["test_sleep"])
        else:
            test_sleep = 1.0
        for i in range(0,test_num):    
            for inp in rc.get_inputs_as_astrodata():
                mtd,nowsec = mock_metadata(inp)
                num =  sin(nowsec/60/60)*.2
                ch = random.choice(["iq", "cc", "bg"])
                if ch == "iq":
                    qad = {"band": 85,
                        "delivered": 0.983,
                        "delivered_error": 0.1,
                        "zenith": 0.7 + num*.4 ,#0.947,
                        "ellipticity": 0.118,
                        "ellip_error": 0.067,
                        "requested": 85,
                        "comment": ["High ellipticity"],
                        }
                elif ch == "cc":
                    qad = {"band": 70,
                            "zeropoint": {"e2v 10031-23-05, right":{"value":26.80,
                                                                    "error":0.05},
                                          "e2v 10031-01-03, right":{"value":26.86,
                                                                    "error":0.03},
                                          "e2v 10031-01-03, left":{"value":26.88,
                                                                   "error":0.06}},
                            "extinction": .5 + num *.5,#0.02,
                            "extinction_error": 0.5,
                            "requested": 50,
                            "comment": ["Requested CC not met"],
                            }
                elif ch == "bg":
                    qad =  {"band": 100,
                            "brightness": 20 + num *.8,#19.17,
                            "brightness_error": 0.5,
                            "requested": 100,
                            "comment": []
                            }
                #print "pG108:"+pprint.pformat(qad)

                rc.report_qametric(inp, ch, qad, metadata = mtd)
            if i%test_burst == 0:
                yield rc
                sleep (test_sleep)
            yield rc
