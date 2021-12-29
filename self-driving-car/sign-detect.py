# Import socket module
import time
import socket
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
import h5py
from scipy.stats import itemfreq
global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
sp = 0
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

def get_dominant_color(image, n_colors):
	pixels = np.float32(image).reshape((-1, 3))
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	flags, labels, centroids = cv2.kmeans(
		pixels, n_colors, None, criteria, 10, flags)
	palette = np.uint8(centroids)
	return palette[np.argmax(itemfreq(labels)[:, -1])]
def bien_cam(image):
	do=0
	img = cv2.resize(src=image, dsize=(1500, 750))
	img = cv2.GaussianBlur(img, (5, 5), 5)
	img = cv2.GaussianBlur(img, (5, 5), 5)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(hsv, (165,60 , 0), (180, 170, 255))
	mask2 = cv2.inRange(hsv, (100,40,30), (140, 120, 60))
	mask = cv2.bitwise_or(mask1, mask2)
	mask_0 = np.zeros_like(img,dtype="uint8")
	img = cv2.bitwise_not(mask_0)

	frame = cv2.bitwise_and(img,img,mask=mask)
	frame = cv2.resize(src=frame, dsize=(320, 160))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(gray, 5)
	
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
							 1, 50, param1=80, param2=9,minRadius=7,maxRadius=30)

	if not circles is None:
		circles = np.uint16(np.around(circles))
		max_r, max_i = 0, 0
		for i in range(len(circles[:, :, 2][0])):
			if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
				max_i = i
				max_r = circles[:, :, 2][0][i]
		x, y, r = circles[:, :, :][0][0]
		if x>r and y>r:
			square = frame[y-r:y+r, x-r:min(x+r,320)]
		elif y>r:
			square = frame[y-r:y+r, 0:x+r]
		elif x>r:
			square = frame[0:y+r, x-r:min(x+r,320)]
		else:
			square = frame[0:y+r, 0:min(x+r,320)]
		zone_0 = square[square.shape[0]*2//8:square.shape[0]*4//8, 
		square.shape[1]*2//8:square.shape[1]*4//8]
		zone_0_color = get_dominant_color(zone_0, 1)

		zone_1 = square[square.shape[0]*4//8:square.shape[0]* 6//8, 
						square.shape[1]*2//8:square.shape[1]*4//8]
		zone_1_color = get_dominant_color(zone_1, 1)

		zone_2 = square[square.shape[0]*4//8:square.shape[0]* 6//8,
					 square.shape[1]*4//8:square.shape[1]*6//8]
		zone_2_color = get_dominant_color(zone_2, 1)

		zone_3 = square[square.shape[0]*2//8:square.shape[0]*4//8, 
					square.shape[1]*4//8:square.shape[1]*6//8]
		zone_3_color = get_dominant_color(zone_3, 1)

		z0 = zone_0_color[0]
		z1 = zone_1_color[0]
		z2 = zone_2_color[0]
		z3 = zone_3_color[0]
		print(z0,z1,z2,z3)
		if z3>30 and z1<30:
			print("cam                            TRAI")
			do = 1
		elif z1>30 and z3>30:
			print("cam PHAI")
			do = 2
		elif z3<60 or z1<60 :
			print("cam            THANG")
			do = 3
		else:
			do = 0
	return do
def bien_xanh(image):
	xanh =0
	img = cv2.resize(src=image, dsize=(1500, 750))
	img = cv2.GaussianBlur(img, (5, 5), 5)
	img = cv2.GaussianBlur(img, (5, 5), 5)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(hsv, (105,120 , 150), (115, 200, 260))
	mask_0 = np.zeros_like(img,dtype="uint8")
	img = cv2.bitwise_not(mask_0)

	frame = cv2.bitwise_and(img,img,mask=mask1)
	frame = cv2.resize(src=frame, dsize=(320, 160))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(gray, 5)

	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
							 1, 50, param1=100, param2=12,minRadius=10,maxRadius=200)

	if not circles is None:
		circles = np.uint16(np.around(circles))
		max_r, max_i = 0, 0
		for i in range(len(circles[:, :, 2][0])):
			if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
				max_i = i
				max_r = circles[:, :, 2][0][i]
		x, y, r = circles[:, :, :][0][0]
		if x>r and y>r:
			square = frame[y-r:y+r, x-r:min(x+r,320)]
		elif y>r:
			square = frame[y-r:y+r, 0:x+r]
		elif x>r:
			square = frame[0:y+r, x-r:min(x+r,320)]
		else:
			square = frame[0:y+r, 0:min(x+r,320)]
		zone_0 = square[square.shape[0]*2//8:square.shape[0]*4//8, 
		square.shape[1]*2//8:square.shape[1]*4//8]
		zone_0_color = get_dominant_color(zone_0, 1)

		zone_1 = square[square.shape[0]*4//8:square.shape[0]* 6//8, 
						square.shape[1]*2//8:square.shape[1]*4//8]
		zone_1_color = get_dominant_color(zone_1, 1)

		zone_2 = square[square.shape[0]*4//8:square.shape[0]* 6//8,
					 square.shape[1]*4//8:square.shape[1]*6//8]
		zone_2_color = get_dominant_color(zone_2, 1)

		zone_3 = square[square.shape[0]*2//8:square.shape[0]*4//8, 
					square.shape[1]*4//8:square.shape[1]*6//8]
		zone_3_color = get_dominant_color(zone_3, 1)
		z0 = zone_0_color[0]
		z1 = zone_1_color[0]
		z2 = zone_2_color[0]
		z3 = zone_3_color[0]
		if z3<200 and z2>150 and z1 < 150:
			print("PHAI")
			xanh = 1
		elif z0<200 and z1>150 and z2<150:
			print("             TRAI")
			xanh = 2
		elif z2<200 and z0<200 and (z1>230 or z3>230):
			if xanh ==1 or xanh ==2:
				pass
			else:
				print("di ve ben PHAI")
				xanh =3
		elif z1<200 and z3<200 and(z0>230 or z2>230 ) :
			if xanh ==1 or xanh ==2:
				pass
			else:
				print("di ve ben TRAI")
				xanh = 4
		elif (z0+z1+z2+z3)<650:
			if xanh ==1 or xanh ==2:
				xanh =0
			
			else:
				print("DI THANG")
				xanh = 5
		else:
			xanh =0

	return xanh

def Control(angle, speed):
	global sendBack_angle, sendBack_Speed
	sendBack_angle = angle
	sendBack_Speed = speed


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument(
		'model',
		type=str,
		help='Path to model h5 file. Model should be on the same path.')
	parser.add_argument(
		'image_folder',
		type=str,
		nargs='?',
		default='',
		help='Path to image folder. This is where the images from the run will be saved.' )
	args = parser.parse_args()
	model = load_model(args.model)
	try:
		while True:
			message_getState = bytes("0", "utf-8")
			s.sendall(message_getState)
			state_date = s.recv(100)

			try:
				current_speed, current_angle = state_date.decode(
					"utf-8"
					).split(' ')
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
				i=bien_cam(image)
				k=bien_xanh(image)
				print(i,k)
				if i == 1:
					polygons = np.array([
							[(50, 160), (320, 160), (320, 60),(160, 60)]
							])
				elif i ==2:
					polygons = np.array([
							[(0, 160), (270, 160), (160, 60),(0, 60)]
							])
				elif i ==3:
					polygons = np.array([
							[(0, 160), (320, 160), (320, 100),(0, 100)]
							])
				elif k ==1:
					polygons = np.array([
							[(50, 160), (320, 160), (320, 60),(160, 60),(90,110)]
							])
				elif k ==2:
					polygons = np.array([
							[(0, 160), (270, 160),(200,110) (160, 60),(0, 60)]
							])
				elif k ==3:
					polygons = np.array([
							[(50, 160), (320, 160), (320, 60),(120, 60)]
							])
				elif k ==4:
					polygons = np.array([
							[(0, 160), (270, 160), (200, 60),(0, 60)]
							])
			
				else:
					polygons = np.array([
							[(0, 160), (320, 160), (320, 60),(0, 60)]
							]) 

					# Fill poly-function deals with multiple polygo
				cv2.fillPoly(mask_0, polygons, [0,255,255]) 
				image = cv2.bitwise_and(img, mask_0)
				image_array = np.asarray(image)
				ag = 25*float(model.predict(image_array[None], batch_size=1))
				pre_angle = abs(float(current_angle)) 
				cv2.imshow('',image)
				cv2.waitKey(1)
				Control(ag, 40)
			except Exception as er:
				print(er)
				pass

	finally:
		print('closing socket')
		s.close()
