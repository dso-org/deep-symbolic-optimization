
import sys
import time

from dsr_batch import DSRBatch
from os import path


#--- use:  python batch.py batchconfigfile

def main():

    stime = time.time()
    print(" Reading parameters ... ",end="")
    sys.stdout.flush()
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    print(" Finished!")
    sys.stdout.flush()
    print(" Generating configuration files ... ",end="")
    DSRBatch(sys.argv)()
    sys.stdout.flush()
    strtime = (" finished! Time: %.1fs" %(time.time() - stime))
    print(strtime)
    sys.stdout.flush()

    return 0


if __name__ == "__main__":

    main()





