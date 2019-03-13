from networktables import NetworkTables
import subprocess

NetworkTables.initialize(server='10.68.29.2')

sd = NetworkTables.getTable('shutdownJetson')

while True:
   shutdown = sd.getBoolean('shutdown', False)
   if shutdown is True:
       subprocess.call("/bin/bash /home/6829/vision/bash/shutdown.sh")