# this is specifically written for the three noisy MNIST case
# might not generalize!

import numpy as np
import theano
from scipy.io import loadmat


# assume these files exist in the same location as this script and all others
f_path = 'path to views data'
mat_filenames = [f_path+'view_b.mat', f_path+'view_h.mat',f_path+'view_l.mat',f_path+'view_p.mat',f_path+'view_s.mat', f_path+'view_v.mat']


def load_noisy_mnist_data(mat_filenames=mat_filenames):
	"""
	matfiles have train_x,y and test_x,y need to generate validation set
	"""
	data_list = []
	for mat_file_ in mat_filenames:
		m_dict = loadmat(mat_file_)
		train_set = make_numpy_array(m_dict['train_data'], np.argmax(m_dict['train_label'], axis=1))
		val_set =  make_numpy_array(m_dict['train_data'], np.argmax(m_dict['train_label'], axis=1))
		test_set = make_numpy_array(m_dict['test_data'], np.argmax(m_dict['test_label'], axis=1))

		data_list.append([train_set, val_set, test_set])
	return data_list

def make_numpy_array(data_x, data_y):
	"""converts the input to numpy arrays"""
	data_x = np.asarray(data_x, dtype=theano.config.floatX)
	data_y = np.asarray(data_y, dtype='int32')
	return (data_x, data_y)
