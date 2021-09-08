"""
Finch dataset with train, test, and validation splits
"""
__date__ = "January - September 2020"

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader



def get_loaders(seed=42, n_folds=7, fold_num=0, nb_fn='juvi1/nb_data.npy', \
	shuffle_control=False, shuffle_seed=43, verbose=True):
	"""
	Get DataLoaders. One fold is test, one is valid, the rest train.

	Parameters
	----------
	seed : int or ``None``, optional
	n_folds : int

	Returns
	-------
	loaders : dict
		Maps the keys 'train', 'test', and 'valid' to respective DataLoaders.
	"""
	fold_num = fold_num % n_folds
	# Load paired neural and behavioral data.
	data = np.load(nb_fn, allow_pickle=True).item()
	# Shuffle.
	np.random.seed(seed)
	perm = np.random.permutation(len(data['fn']))
	np.random.seed(None)
	for key in ['fn', 'calcium', 'spec', 'tempo']:
		data[key] = data[key][perm]
	# Collect data.
	fns = data['fn']
	times = get_times(data)
	# Define train/valid/test indices.
	idx = [int(round(i)) for i in np.linspace(0,len(times),n_folds+1)]
	chunks = [np.arange(idx[i],idx[i+1]) for i in range(n_folds)]
	test_index = chunks[fold_num]
	valid_index = chunks[(fold_num+1) % n_folds]
	train_index = [chunks[i] for i in range(n_folds) if i != fold_num and i != (fold_num+1)%n_folds]
	train_index = np.concatenate(train_index)
	# Optional: shuffle to make sure we're not fooling ourselves.
	if shuffle_control:
		np.random.seed(shuffle_seed)
		perm = np.random.permutation(len(data['calcium']))
		np.random.seed(None)
		data['calcium'] = data['calcium'][perm]
	# Get all the data set.
	train_n = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['calcium'][train_index]])
	train_b = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['spec'][train_index]]).view(-1,128*128)
	train_fns = fns[train_index]
	train_times = times[train_index]
	train_tempo = None
	valid_n = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['calcium'][valid_index]])
	valid_b = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['spec'][valid_index]]).view(-1,128*128)
	valid_fns = fns[valid_index]
	valid_times = times[valid_index]
	valid_tempo = None
	test_n = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['calcium'][test_index]])
	test_b = torch.stack([torch.tensor(i, dtype=torch.float) for i in data['spec'][test_index]]).view(-1,128*128)
	test_fns = fns[test_index]
	test_times = times[test_index]
	test_tempo = None
	# Check if tempo is recorded.
	if 'tempo' in data:
		train_tempo = np.array(data['tempo'])[train_index]
		valid_tempo = np.array(data['tempo'])[valid_index]
		test_tempo = np.array(data['tempo'])[test_index]
	# Print stuff.
	if verbose:
		print("Making FinchDatasets:")
		print("\tvalid:", len(valid_n))
		print("\ttrain:", len(train_n))
		print("\ttest:", len(test_n))
		print("\tseed:", seed)
		print("\tn_folds:", n_folds)
		print("\tfold_num:", fold_num)
		if shuffle_control:
			print("\tSHUFFLE CONTROL!")
	# Make datasets.
	train_dset = FinchDataset(neural=train_n, behavioral=train_b, fns=train_fns, times=train_times, tempo=train_tempo)
	valid_dset = FinchDataset(neural=valid_n, behavioral=valid_b, fns=valid_fns, times=valid_times, tempo=valid_tempo)
	test_dset = FinchDataset(neural=test_n, behavioral=test_b, fns=test_fns, times=test_times, tempo=test_tempo)
	train_valid_dset = train_dset + valid_dset
	# Make loaders.
	train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
	valid_loader = DataLoader(valid_dset, batch_size=64, shuffle=False)
	test_loader = DataLoader(test_dset, batch_size=64, shuffle=False)
	train_valid_loader = DataLoader(train_valid_dset, batch_size=64, shuffle=True)
	return {
			'valid': valid_loader,
			'train': train_loader,
			'test': test_loader,
			'train_valid': train_valid_loader,
	}


def get_single_loader(nb_fn='juvi1/nb_data.npy', shuffle_control=False, shuffle_seed=43):
	"""Put all the data in a single loader."""
	data = np.load(nb_fn, allow_pickle=True).item()
	if shuffle_control:
		np.random.seed(shuffle_seed)
		perm = np.random.permutation(len(data['calcium']))
		np.random.seed(None)
		data['calcium'] = data['calcium'][perm]
	fns = data['fn']
	times = get_times(data)
	calcium = torch.tensor(data['calcium'], dtype=torch.float)
	spec = torch.tensor(data['spec'], dtype=torch.float).view(-1,128*128)
	tempo = None
	if 'tempo' in data:
		tempo = np.array(data['tempo'])
	dset = FinchDataset(neural=calcium, behavioral=spec, fns=fns, times=times, tempo=tempo)
	loader = DataLoader(dset, batch_size=64, shuffle=True)
	return {'train': loader}


def get_simple_test_train_loader(nb_fn='juvi1/nb_data.npy', seed=42, train_portion=0.8):
	"""Put all the data in a single loader."""
	data = np.load(nb_fn, allow_pickle=True).item()
	np.random.seed(seed)
	perm = np.random.permutation(len(data['calcium']))
	np.random.seed(None)
	calcium = data['calcium'][perm]
	spec = data['spec'][perm]
	fns = data['fn'][perm]
	times = get_times(data)[perm]
	idx = int(round(train_portion * len(times)))
	train_calcium = torch.tensor(calcium[:idx], dtype=torch.float)
	train_spec = torch.tensor(spec[:idx], dtype=torch.float).view(-1,128*128)
	train_times = times[:idx]
	train_fns = fns[:idx]
	test_calcium = torch.tensor(calcium[idx:], dtype=torch.float)
	test_spec = torch.tensor(spec[idx:], dtype=torch.float).view(-1,128*128)
	test_times = times[idx:]
	test_fns = fns[idx:]
	train_tempo, test_tempo = None, None
	if 'tempo' in data:
		train_tempo = np.array(data['tempo'])[perm][:idx]
		test_tempo = np.array(data['tempo'])[perm][idx:]
	train_dset = FinchDataset(neural=train_calcium, behavioral=train_spec, \
			fns=train_fns, times=train_times, tempo=train_tempo)
	train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
	test_dset = FinchDataset(neural=test_calcium, behavioral=test_spec, \
			fns=test_fns, times=test_times, tempo=test_tempo)
	test_loader = DataLoader(test_dset, batch_size=64, shuffle=True)
	return {'train': train_loader, 'test': test_loader}


def get_times(data):
	"""Extract motif times from the .npy file, z-score, return."""
	fns = data['fn']
	onsets = data['onset']
	times = np.zeros(len(fns))
	for i in range(len(fns)):
		hour = float(fns[i][-10:-8])
		minute = float(fns[i][-8:-6])
		second = float(fns[i][-6:-4]) + onsets[i]
		times[i] = 60*60*hour + 60*minute + second
	times -= np.mean(times)
	times /= np.std(times)
	return times



class FinchDataset(Dataset):

	def __init__(self, neural=None, behavioral=None, fns=None, times=None, tempo=None):
		"""
		Finch data.

		Pass neural and behavioral as float tensors.
		"""
		assert neural is not None and behavioral is not None
		assert fns is not None and times is not None
		self.neural = neural
		self.behavioral = behavioral
		self.fns = fns
		self.times = times
		self.tempo = tempo
		self.has_tempo = tempo is None


	def __len__(self):
		return self.neural.shape[0]


	def __getitem__(self, index, rand_perm=True):
		d = {
			'n': self.neural[index],
			'b': self.behavioral[index],
			'fn': self.fns[index],
			'time': self.times[index],
			'index': index,
		}
		if self.has_tempo:
			d['tempo'] = self.tempo[index]
		return d


	def __str__(self):
		repr = "FinchDataset, neural: "
		repr += str(self.neural.shape)
		repr += ", behavioral: "
		repr += str(self.behavioral.shape)
		return repr


	def __add__(self, other):
		neural = torch.cat([self.neural, other.neural], dim=0)
		behavioral = torch.cat([self.behavioral, other.behavioral], dim=0)
		fns = np.concatenate([self.fns, other.fns], axis=0)
		times = np.concatenate([self.times, other.times], axis=0)
		if self.tempo is None or other.tempo is None:
			tempo = None
		else:
			tempo = np.concatenate([self.tempo, other.tempo], axis=0)
		return FinchDataset(neural=neural, behavioral=behavioral, fns=fns, \
				times=times, tempo=tempo)



if __name__ == '__main__':
	pass



###
