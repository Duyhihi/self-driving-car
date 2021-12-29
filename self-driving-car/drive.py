# Import socket module
import socket
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
import h5py

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
			message_getState = bytes("0", 'utf-8')
			s.sendall(message_getState)
			state_date = s.recv(100)

			try:
				current_speed, current_angle = state_date.decode(
					'utf-8'
					).split(' ')
			except Exception as er:
				print(er)
				pass

			message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", 'utf-8')
			s.sendall(message)
			data = s.recv(100000)

			try:
				image = cv2.imdecode(
					np.frombuffer(
						data,
						np.uint8
						), -1
					)
				cv2.imshow("IMG", image)
				cv2.waitKey(1)
				img_crop = image[170:360, 0:640, :]
				image = cv2.resize(src=img_crop, dsize=(320, 160))
				image_array = np.asarray(image)
				ag = 25*float(model.predict(image_array[None], batch_size=1))
				pre_angle = abs(float(current_angle)) 
				print(current_speed, current_angle)
				print(image.shape)
				
				Control(ag, 50)

			except Exception as er:
				print(er)
				pass

	finally:
		print('closing socket')
		s.close()
