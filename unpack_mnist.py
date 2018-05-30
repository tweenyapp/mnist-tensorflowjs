import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt
from scipy.misc import imsave

train_data = ['mnist_data/train-images-idx3-ubyte.gz', 'mnist_data/train-labels-idx1-ubyte.gz']
test_data = ['mnist_data/t10k-images-idx3-ubyte.gz', 'mnist_data/t10k-labels-idx1-ubyte.gz']

def load_data_labels(path):
	imgs, labels_list = [], []
	for i,file_ in enumerate(path):
		with gzip.open(file_, 'rb') as f:
			if i == 0:
				magic, size, width, height = struct.unpack('>4i', f.read(16))
				chunk = width * height
				for _ in range(size):
					img = struct.unpack('>%dB' % chunk, f.read(chunk))
					img_np = np.array(list(img))#.reshape(width, height)					
					imgs.append(img_np)

			else:
				magic, size = struct.unpack('>2i', f.read(8))
				for label in struct.unpack('>%dB' % size, f.read()):
					labels_list.append(label)

	return np.array(imgs), labels_list

imgs_train, labels_train = load_data_labels(train_data)
for i in range(20):
	imsave('mnist_data/mnist_batch_'+str(i)+'.png', imgs[i*3000:3000*(i+1),:])

imgs_test, labels_test = load_data_labels(test_data)
imsave('mnist_data/mnist_batch_'+str(20)+'.png', imgs_test)

with open('mnist_labels.js', 'a') as f:
	L = 'var trainLabels=' + str(labels_train) + ';\n'
	f.write(L)
	L = 'var testLabels=' + str(labels_test) + ';\n'
	f.write(L)