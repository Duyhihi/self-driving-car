import cv2
import numpy as np
import random
import matplotlib.image as mpimg
import pandas as pd
import glob 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.optimizers import Adam		
import matplotlib.pyplot as plt


IMG_H,IMG_W,IMG_CH=160,320,3
def augment(image, measurement):
	#đảo dữ liệu ngẫu nhiên
	if np.random.rand() < 0.5:
		image = np.fliplr(image)
		measurement = -measurement
	return image, measurement

def load_data(test_size):
	#nạp data 
	names = ['center','speed','steering']
	data_df = pd.read_csv('driving_log.csv',header=None,names=names)
	X = data_df['center'].values
	y = data_df['steering'].values
	return train_test_split(X,y,test_size=test_size) # chia thành các mẫu test và train :D



def build_model():

	model = Sequential()
	model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(IMG_H,IMG_W,IMG_CH)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def batch_generator(image_paths,steering_angles,batch_size):
	#lấy batch_size data ngẫu nhiên để train :D làm lại n lần 
	images = np.empty([batch_size,IMG_H,IMG_W,IMG_CH])
	measurements = np.empty(batch_size)
	while True:
		i=0
		for index in np.random.permutation(image_paths.shape[0]):
			center = image_paths[index]
			image = mpimg.imread(center)
			measurement = float(steering_angles[index])/25
			image,measurement = augment(image,measurement)
			images[i] = image
			measurements[i] = measurement
			i+=1
			if i== batch_size:
				break
		yield images, measurements



def main():
	test_size = 0.2
	batch_size=64
	epochs =5
	verbose =1
	print('Loading data....')
	X_train,X_val,y_train,y_val = load_data(test_size)
	print('Building models...')
	model = build_model()
	Name=1
	

	print('Compiling model...')
	model.compile(loss='mse', optimizer=Adam(lr=0.0001))
	print('Training...')
	history_object = model.fit_generator(batch_generator(X_train,y_train,batch_size),
										steps_per_epoch=len(X_train)/batch_size,#làm lại bằng số mẫu chia số batch 
										validation_data=batch_generator(X_val,y_val,batch_size),
										validation_steps=len(X_val)/batch_size,
										epochs=epochs,
										verbose=verbose)
	print('Saving models...')
	model.save('model.h5')
	print('Model saved')
	print(history_object.history.keys())
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper left')
	plt.savefig('history.png',bbox_inches='tight')

if __name__ == '__main__':
	main()