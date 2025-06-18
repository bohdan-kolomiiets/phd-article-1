from datetime import datetime

def printd(*args, **kw):
    print("[%s]" % (datetime.now()),*args, **kw)