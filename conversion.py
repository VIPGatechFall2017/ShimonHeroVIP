
preData = open(r'C:\Users\ejen0\OneDrive\Documents\Fall 2017\VIP\ShimonHeroVIP\data.txt','r')
file = open("convertedData.txt",'w')	

for i in preData.readlines():
	lineOfData = i.split(" ")
	lineOfData[0] = int(lineOfData[0][0:-1])
	if int(lineOfData[1]) > 60:
		lineOfData[1] = 1
	else:
		lineOfData[1] = 0
	if int(lineOfData[2]) > 60:
		lineOfData[2] = 1
	else:
		lineOfData[2] = 0
	if int(lineOfData[3]) > 60:
		lineOfData[3] = 1
	else:
		lineOfData[3] = 0
	if int(lineOfData[4]) > 60:
		lineOfData[4] = 1
	else:
		lineOfData[4] = 0
	lineOfData[5] = int(lineOfData[5][0:-2])
	if int(lineOfData[5]) > 60:
		lineOfData[5] = 1
	else:
		lineOfData[5] = 0
	file.write(str(lineOfData) + "\n")

preData.close()
file.close()

