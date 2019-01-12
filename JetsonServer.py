from networktables import NetworkTables

NetworkTables.initialize(server='10.68.29.2');

sd = NetworkTables.getTable('Vision')

val = 0

while True:
	val+=1
	print("val " + str(val))
	sd.putNumber('message', val)
