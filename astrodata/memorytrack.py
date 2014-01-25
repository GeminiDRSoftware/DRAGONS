MEMTRACK = True

def memtrack(primname = "unknown_point", msg = "", context = None):
    if MEMTRACK:
        import psutil
        import os
        import json
        import time
        import gc
        
        gc.collect()
        tpid = os.getpid()
        mfile = open("memtrack-%d" % os.getpid(), "a+");
        proc = psutil.Process(tpid)
        memi = proc.get_ext_memory_info()
        item = {    "msg": "%s [%s]" %(primname, msg),
                    "primname": primname,
                    "rss": memi.rss,
                    "vms": memi.vms,
                    "pfaults": memi.pfaults,
                    "pageins": memi.pageins,
                    "time": time.time()
                }
        
        line = json.dumps(item)+"\n"
        mfile.write(line)
        mfile.close()    