import time
import threading
import datetime

smpPeriod = 1     #sample period in sec
#--------------------------------------------------------------------------------------------
# Enable a stable time to triger data collection
#--------------------------------------------------------------------------------------------            
class T0(object):
    
    def __init__(self, period):
        self.pd = period
    
    def run(self):
        ts = datetime.datetime.now()
        while (True):
            tf = datetime.datetime.now()
            print('Delta time: ',  tf-ts)
            ts=tf
            time.sleep(self.pd)

#------------------------------------------------------------------------------------------------        
if __name__ == '__main__':

    t0 = T0(smpPeriod)
    mt = threading.Thread(target = t0.run,  args=())

    mt.start()  # start the triger timer
    
    