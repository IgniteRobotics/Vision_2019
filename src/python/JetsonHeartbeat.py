from networktables import NetworkTables

import datetime
import time

print("Heartbeat started")
NetworkTables.initialize(server='10.68.29.2')

hb = NetworkTables.getTable('HeartBeat')

while True:
   
    ts = datetime.datetime.now().timestamp()
    print(str(ts))
    hb.putString('timestamp', str(ts) )
    
    time.sleep(1.0)
    