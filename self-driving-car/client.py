import random
import socket
import cv2
import numpy as np
import csv
import os
from scipy.stats import itemfreq
scount = 1
path = os.getcwd()
global sendBack_angle, sendBack_Speed, current_speed, current_angle,angle_change,speed_change
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
count = 0
r=0
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

def Control(angle, speed):
	global sendBack_angle, sendBack_Speed
	sendBack_angle = angle
	sendBack_Speed = speed

with open('./driving_log.csv','w',newline='') as f:
	writer = csv.writer(f)
	if __name__ == "__main__":
		try:
			while True:

				message_getState = bytes("0", "utf-8")
				s.sendall(message_getState)
				state_date = s.recv(100)

				try:
					current_speed, current_angle = state_date.decode("utf-8").split(' ')
				except Exception as er:
					print(er)
					pass

								
				message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
				s.sendall(message)
				data = s.recv(100000)

				try:
					image = cv2.imdecode(
						np.frombuffer(
							data,
							np.uint8
							), -1
						)
					image = cv2.resize(src=image, dsize=(320, 160))
					print(current_speed, current_angle)
					print(image.shape)
					img = cv2.GaussianBlur(image, (5, 5), 5)
					img = cv2.GaussianBlur(img, (5, 5), 5)

					hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
					mask1 = cv2.inRange(hsv, (30,0 , 100), (100, 20, 200))
					mask2 = cv2.inRange(hsv, (103,50,70), (110, 120, 120))
					mask3 = cv2.inRange(hsv, (100,20,75), (110, 60, 150))
					mask = cv2.bitwise_or(mask1, mask2)
					mask = cv2.bitwise_or(mask, mask3)
					mask_0 = np.zeros_like(img,dtype="uint8")
					img = cv2.bitwise_not(mask_0)
					img = cv2.bitwise_and(img,img,mask=mask)
					polygons = np.array([
							[(0, 160), (320, 160), (320, 60),(0, 60)]
							])
						# Fill poly-function deals with multiple polygo
					cv2.fillPoly(mask_0, polygons, [0,255,255]) 
					image = cv2.bitwise_and(img, mask_0) 
					if (float(current_speed)>20) and ((0.03>random.random()) or (current_angle!="0.00")): #chỉ lấy giá trị khi góc và vận tốc khác 0 để giảm mẫu
						cv2.imwrite("IMG/frame%d.jpg" % count, image) 
						link_image = os.path.join(path, "IMG", "frame%d.jpg" % count)
						writer.writerow([link_image, current_speed, current_angle])
						count += 1
						cv2.imshow('',image)
						cv2.waitKey(1)
												

				except Exception as er:
					print(er)
					pass

		finally:
			print('closing socket')
			s.close()
