"""
Run model performance experiments.

"""
__date__ = "May - December 2020"

from itertools import product
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
import warnings

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression

from ssl.finch_data import get_loaders, get_single_loader, get_simple_test_train_loader
from ssl.poe_finch import PoeFinch
from ssl.cca_finch import CCAFinch
from ssl.linear_cca_finch import CCAFinch as LinearCCAFinch
from ssl.linear_poe_finch import PoeFinch as LinearPoeFinch

BIRD, DATA_DIR, NB_FN, N_NEURONS = None, None, None, None
ALL_BIRDS = ['JUVI1_0305', 'JUVI1_0228', 'BLK215_1005', 'BLK215_0929', \
		'BLK202', 'ORG137', 'YEL141']

def set_bird(bird):
	global BIRD, DATA_DIR, NB_FN, N_NEURONS
	BIRD = bird
	if BIRD == 'JUVI1_0305':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
		NB_FN = os.path.join(DATA_DIR, 'nb_20180305_spec_calcium.npy')
		N_NEURONS = 76
	elif BIRD == 'JUVI1_0228':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
		NB_FN = os.path.join(DATA_DIR, 'nb_20180228_spec_calcium.npy')
		N_NEURONS = 41
	elif BIRD == 'BLK215_1005':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
		NB_FN = os.path.join(DATA_DIR, 'nb_10052018_spec_calcium.npy')
		N_NEURONS = 70
	elif BIRD == 'BLK215_0929':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
		NB_FN = os.path.join(DATA_DIR, 'nb_09292018_spec_calcium.npy')
		N_NEURONS = 44
	elif BIRD == 'BLK202':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk202/'
		NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
		N_NEURONS = 44
	elif BIRD == 'ORG137':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/org137/'
		NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
		N_NEURONS = 78
	elif BIRD == 'YEL141':
		DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/yel141/'
		NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
		N_NEURONS = 36
	else:
		raise NotImplementedError

ALPHA_VALS = np.geomspace(1e-4, 1e5, 19)
GAMMA_VALS = np.geomspace(1e-4,1e2, 13)
RIDGE_TOL = 1e-3


exp_params = {
	'poe_finch': {
		'suffix': '_poe_finch_',
		'label': 'Joint Encoding',
		'c': 'darkorchid',
		'model': PoeFinch,
	},
	'cca_finch': {
		'suffix': '_cca_finch_',
		'label': 'Separate Encoding',
		'c': 'mediumseagreen',
		'model': CCAFinch,
	},
	'linear_cca_finch': {
		'suffix': '_linear_cca_finch_',
		'label': 'Separate Encoding',
		'c': 'mediumseagreen',
		'model': LinearCCAFinch,
	},
	'linear_poe_finch': {
		'suffix': '_linear_poe_finch_',
		'label': 'Joint Encoding',
		'c': 'darkorchid',
		'model': LinearPoeFinch,
	},
}



def get_res_model_filenames(model_type, model_num):
	"""Get some filenames for a given model."""
	suffix = exp_params[model_type]['suffix'] + str(model_num).zfill(2)
	res_fn = os.path.join(DATA_DIR, 'res' + suffix + '.npy')
	model_fn = os.path.join(DATA_DIR, 'net' + suffix + '.tar')
	return res_fn, model_fn


def cross_validation_run(model_type, model_num, spec_stds, calcium_stds, lrs, \
	objective='spec_l_to_calcium_l', max_epochs=1000, \
	epochs_with_no_improvement=200, print_freq=20, valid_freq=10, \
	shuffle_control=False, verbose=True):
	"""
	Perform a full cross validation run, get an R^2 estimate on the test set.

	In more detail: Train a model for every combination of spec_std,
	calcium_std, and lr. Every so often, evaluate the objective on the
	validation set. If this is the best result seen so far, also evaluate on the
	test set. Return the evaluation on the test set corresponding to the best
	evaluation on the validation set.

	Parameters
	----------

	Returns
	-------

	"""
	assert objective in ['spec_l_to_calcium_l', 'calcium_l_to_spec_l']
	if verbose:
		print(model_type)
	loaders = get_loaders(nb_fn=NB_FN, fold_num=model_num, \
			shuffle_control=shuffle_control, verbose=verbose)
	res_fn, model_fn = get_res_model_filenames(model_type, model_num)
	best_items = {
			'epoch': -1,
			'valid_obj': -np.inf,
			'test_obj': -np.inf,
			'time_obj_linear': -np.inf,
			'tempo_obj_linear': -np.inf,
			'time_obj_kernel': -np.inf,
			'tempo_obj_kernel': -np.inf,
			'valid_params': None,
			'lr': -1,
			'calcium_std': -1,
			'spec_std': -1,
	}
	for spec_std, calcium_std, lr in product(spec_stds, calcium_stds, lrs):
		if verbose:
			print("spec_std:", spec_std, "calcium_std:", calcium_std, "lr:", lr)
		model = exp_params[model_type]['model'](model_type=model_type, \
				calcium_dim=N_NEURONS, spec_std=spec_std, \
				calcium_std=calcium_std, lr=lr)
		# Track the best within-model performance for early stopping criterion.
		model_best_items = {'epoch':-1, 'valid_obj':-np.inf}
		# Train for some number of epochs.
		for epoch in range(max_epochs):
			loss, _ = model.train_epoch(loaders['train'])
			if verbose and epoch > 0 and epoch % print_freq == 0:
				print("Epoch:", str(epoch).zfill(4), round(loss,6))
			if epoch % valid_freq == 0:
				with torch.no_grad():
					# Gather data.
					valid_latent = model.get_local_latents(loaders['valid'])
					train_latent = model.get_local_latents(loaders['train'])
					if objective == 'spec_l_to_calcium_l':
						train_X, train_Y = train_latent['spec'], train_latent['calcium']
						valid_X, valid_Y = valid_latent['spec'], valid_latent['calcium']
					elif objective == 'calcium_l_to_spec_l':
						train_X, train_Y = train_latent['calcium'], train_latent['spec']
						valid_X, valid_Y = valid_latent['calcium'], valid_latent['spec']
					else:
						raise NotImplementedError
					# Evaluate objective on validation set.
					# valid_r2 = run_lko_ridge_regression(train_X, train_Y, valid_X, valid_Y)
					valid_r2, valid_params = \
						run_ridge_regression(train_X, train_Y, valid_X, valid_Y)
					# If it's the best we've seen in this run, update some stats.
					if valid_r2 > model_best_items['valid_obj']:
						model_best_items['valid_obj'] = valid_r2
						model_best_items['epoch'] = epoch
					# If it's the best we've seen ever, evaluate on the test set.
					if valid_r2 > best_items['valid_obj']:
						# Get test latent.
						test_latent = model.get_local_latents(loaders['test'])
						if objective == 'spec_l_to_calcium_l':
							test_X, test_Y = test_latent['spec'], test_latent['calcium']
						elif objective == 'calcium_l_to_spec_l':
							test_X, test_Y = test_latent['calcium'], test_latent['spec']
						else:
							raise NotImplementedError
						# Evaluate on test set.
						test_r2, _ = run_ridge_regression(train_X, train_Y, \
								test_X, test_Y, fixed_params=valid_params)
						# Save parameter settings.
						best_items['valid_obj'] = valid_r2
						best_items['test_obj'] = test_r2
						best_items['valid_params'] = valid_params
						best_items['epoch'] = epoch
						best_items['lr'] = lr
						best_items['calcium_std'] = calcium_std
						best_items['spec_std'] = spec_std
						# Also document how well time predicts.
						train_times = train_latent['times'].reshape(-1,1)
						valid_times = valid_latent['times'].reshape(-1,1)
						test_times = test_latent['times'].reshape(-1,1)
						# Linear time.
						_, time_params = run_ridge_regression(train_times, \
								train_Y, valid_times, valid_Y)
						time_r2, _ = run_ridge_regression(train_times, \
								train_Y, test_times, test_Y, \
								fixed_params=time_params)
						best_items['time_obj_linear'] = time_r2
						# Kernel time.
						_, time_params = run_ridge_regression(train_times, \
								train_Y, valid_times, valid_Y, mode='kernelized')
						time_r2, _ = run_ridge_regression(train_times, \
								train_Y, test_times, test_Y, \
								fixed_params=time_params, mode='kernelized')
						best_items['time_obj_kernel'] = time_r2
						# Also document how well tempo predicts.
						train_tempo = train_latent['tempo'].reshape(-1,1)
						valid_tempo = valid_latent['tempo'].reshape(-1,1)
						test_tempo = test_latent['tempo'].reshape(-1,1)
						# Linear tempo.
						_, tempo_params = run_ridge_regression(train_tempo, \
								train_Y, valid_tempo, valid_Y)
						tempo_r2, _ = run_ridge_regression(train_tempo, \
								train_Y, test_tempo, test_Y, \
								fixed_params=tempo_params)
						best_items['tempo_obj_linear'] = tempo_r2
						# Kernel tempo.
						_, tempo_params = run_ridge_regression(train_tempo, \
								train_Y, valid_tempo, valid_Y, mode='kernelized')
						tempo_r2, _ = run_ridge_regression(train_tempo, \
								train_Y, test_tempo, test_Y, \
								fixed_params=tempo_params, mode='kernelized')
						best_items['tempo_obj_kernel'] = tempo_r2
			if epoch - model_best_items['epoch'] > epochs_with_no_improvement:
				if verbose:
					print("Last improvement:",model_best_items['epoch'],", Epoch", epoch)
					print("Valid obj:", model_best_items['valid_obj'])
					print("Global best:", best_items, "\n")
				break # break out of training loop, on to the next model
		if epoch == max_epochs -1:
			if verbose:
				print("REACHED MAX_EPOCHS!")
				print("Last improvement:",model_best_items['epoch'],", Epoch", epoch)
				print("Valid obj:", model_best_items['valid_obj'])
				print("Global best:", best_items, "\n")
	# Return stuff.
	return best_items


def run_ridge_regression(train_X, train_Y, test_X, test_Y, \
	alpha_vals=ALPHA_VALS, gamma_vals=GAMMA_VALS, mode='linear', \
	fixed_params=None):
	""" """
	assert mode in ['linear', 'kernelized']
	if fixed_params:
		assert alpha_vals is ALPHA_VALS and gamma_vals is GAMMA_VALS
		if mode == 'linear':
			assert 'gamma' not in fixed_params
			alpha_vals = [fixed_params['alpha']]
		else:
			alpha_vals = [fixed_params['alpha']]
			gamma_vals = [fixed_params['gamma']]
	best_r2 = -np.inf
	if mode == 'linear':
		best_params = {'alpha': None}
	else:
		best_params = {'alpha': None, 'gamma': None}
	mean_train_Y = np.mean(train_Y, axis=0, keepdims=True)
	term_1 = np.sum(np.power(test_Y - mean_train_Y, 2))
	for alpha in alpha_vals:
		if mode == 'linear':
			reg = Ridge(alpha=alpha, fit_intercept=True, tol=RIDGE_TOL)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				reg.fit(train_X, train_Y)
			pred_test_Y = reg.predict(test_X)
			term_2 = np.sum(np.power(test_Y - pred_test_Y, 2))
			r2 = 1.0 - term_2 / term_1
			if r2 > best_r2:
				best_r2 = r2
				best_params['alpha'] = alpha
		else:
			for gamma in gamma_vals:
				reg = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					reg.fit(train_X, train_Y)
				pred_test_Y = reg.predict(test_X)
				term_2 = np.sum(np.power(test_Y - pred_test_Y, 2))
				r2 = 1.0 - term_2 / term_1
				if r2 > best_r2:
					best_r2 = r2
					best_params['alpha'] = alpha
					best_params['gamma'] = gamma
	return best_r2, best_params


def run_lko_ridge_regression(train_X, train_Y, test_X, test_Y, \
	alpha_vals=ALPHA_VALS, gamma_vals=GAMMA_VALS, k=1, shuffle=True, \
	mode='linear'):
	"""
	Get an R^2 from a leave-k-out ridge regression.

	Choose a model based on validation set performance, where the validation set
	is everything in the test set except for a rotating k items.
	"""
	assert mode in ['linear', 'kernelized']
	pred_test_Y = np.zeros(test_Y.shape)
	mean_train_Y = np.mean(train_Y, axis=0, keepdims=True)
	# Shuffle the test set.
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(test_X))
		np.random.seed(None)
		test_X = test_X[perm]
		test_Y = test_Y[perm]
	# Run ridge regression on the training set for each alpha value.
	regs = []
	for alpha in alpha_vals:
		if mode == 'linear':
			temp_reg = Ridge(alpha=alpha, fit_intercept=True, tol=RIDGE_TOL)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				temp_reg.fit(train_X, train_Y)
			regs.append(temp_reg)
		else:
			for gamma in gamma_vals:
				temp_reg = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					temp_reg.fit(train_X, train_Y)
				regs.append(temp_reg)
	# Leave out a different k items.
	for i in range(len(test_Y)//k):
		temp_valid_X = np.concatenate([test_X[:i*k], test_X[(i+1)*k:]], axis=0)
		temp_test_X = test_X[i*k:(i+1)*k]
		temp_valid_Y = np.concatenate([test_Y[:i*k], test_Y[(i+1)*k:]], axis=0)
		# See which alpha performs best on the validation set.
		temp_r2s = []
		term_1 = np.sum(np.power(temp_valid_Y - mean_train_Y, 2))
		for j, alpha in enumerate(alpha_vals):
			if mode == 'linear':
				temp_valid_Y_pred = regs[j].predict(temp_valid_X)
				term_2 = np.sum(np.power(temp_valid_Y - temp_valid_Y_pred, 2))
				temp_r2s.append(1.0 - term_2 / term_1)
			else:
				for g in range(len(gamma_vals)):
					temp_valid_Y_pred = regs[j*len(gamma_vals)+g].predict(temp_valid_X)
					term_2 = np.sum(np.power(temp_valid_Y - temp_valid_Y_pred, 2))
					temp_r2s.append(1.0 - term_2 / term_1)
		temp_r2s = np.array(temp_r2s)
		idx = np.argmax(temp_r2s)
		# Predict test_Y on this model.
		pred_test_Y[i*k:(i+1)*k] = regs[idx].predict(temp_test_X)
	# Calculate an overall R^2 value.
	term_1 = np.sum(np.power(test_Y - mean_train_Y,2))
	term_2 = np.sum(np.power(test_Y - pred_test_Y, 2))
	r2 = 1.0 - term_2 / term_1
	return r2



def simple_run(model_type, model_fn, epochs=300, shuffle_control=False, \
	save=True, single_loader=False, save_rec=False):
	""" """
	model = exp_params[model_type]['model'](model_type=model_type, \
			calcium_dim=N_NEURONS, spec_std=0.04, \
			calcium_std=0.2, lr=1e-3, fn=model_fn)
	# model = PoeFinch(model_type='poe_finch', fn=model_fn, calcium_dim=N_NEURONS)
	if single_loader:
		loaders = get_single_loader(nb_fn=NB_FN, shuffle_control=shuffle_control)
	else:
		loaders = get_simple_test_train_loader(nb_fn=NB_FN)
	for epoch in range(epochs):
		loss, _ = model.train_epoch(loaders['train'])
		if epoch % 20 == 0:
			print("Epoch:", str(epoch).zfill(4), loss)
		if epoch % 20 == 0:
			print("plotting")
			model.rec_plot(loaders['train'], save=save_rec)
	if save:
		model.save_state()



if __name__ == '__main__':
	# for bird in ALL_BIRDS:
	# 	set_bird(bird)
	# 	d = np.load(NB_FN, allow_pickle=True).item()
	# 	print(bird, d['calcium'].shape[0], d['calcium'].shape[1])
	# quit()
	#
	# # import sys
	# # print("sys.argv:", sys.argv)
	# # assert len(sys.argv) == 2
	# # index = int(sys.argv[1])
	# # print("Bird:", ALL_BIRDS[index])
	# # set_bird(ALL_BIRDS[index])
	#
	# set_bird('JUVI1_0305')
	# # model_fn = 'blk215_0929_separate_vis_model.tar'
	# simple_run('poe_finch', None, epochs=301, save=False, save_rec=True)
	# quit()

	# 5-hour experiment
	# model_types = ['cca_finch', 'poe_finch', 'linear_cca_finch', 'linear_poe_finch']
	model_types = ['poe_finch']
	objectives = ['calcium_l_to_spec_l', 'spec_l_to_calcium_l']
	spec_stds = [0.02, 0.04]
	calcium_stds = [0.2, 0.4]
	lrs = [1e-3]
	# Run.
	set_bird(ALL_BIRDS[0])
	for model_type in model_types:
		print("\n", model_type)
		for objective in objectives:
			for fold in range(7):
				res = cross_validation_run(model_type, fold, spec_stds, calcium_stds, lrs, \
					print_freq=1000, shuffle_control=False, objective=objective, verbose=False)
				print(model_type, objective, fold, res)



###
