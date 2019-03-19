from networktables import NetworkTables
import subprocess

NetworkTables.initialize(server='10.68.29.2')

sd = NetworkTables.getTable('Launcher')

old_ip_address = '10.68.29.20'

startup_flag = False

while True:
    ip_address = sd.getString('ip_address', 'no')
    if ip_address != old_ip_address:
        old_ip_address = ip_address
        subprocess.call("/usr/bin/pkill -f gst-launch-1.0")
        subprocess.call("/bin/bash /home/nvidia/6829/vision/bash/gstreamer_server.sh /dev/streamcam {} 5805".format(ip_address))
    elif startup_flag == False:
        subprocess.call("/bin/bash /home/nvidia/6829/vision/bash/gstreamer_server.sh /dev/streamcam {} 5805".format(old_ip_address))
        startup_flag = True
