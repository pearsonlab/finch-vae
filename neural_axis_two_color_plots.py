"""
Neural axis two color plots.

Pick an axis in neural space and plot average reconstructions overlayed.

"""
__date__ = "September - November 2020"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import savemat
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
import torch

from ssl.finch_data import get_simple_test_train_loader
from ssl.poe_finch import PoeFinch
from ssl.cca_finch import CCAFinch



def generate_data(model, n_batches=5, samples_per_batch=200):
	"""Generate random calcium/spectrogram pairs."""
	calcium, spec, yn_samples, yb_samples = [], [], [], []
	with torch.no_grad():
		for i in range(n_batches):
			z_sample = torch.randn(samples_per_batch,1,1,model.z_dim).to(model.device)
			yn_sample = model.sample_yn(z_sample)
			yb_sample = model.sample_yb(z_sample)
			c_μ = model.decode_calcium(yn_sample).squeeze()
			if model.model_type == 'sparse_poe_finch':
				spec_rec, _ = model.decode_spec(yb_sample)
			else:
				spec_rec = model.decode_spec(yb_sample)
			spec_rec = spec_rec.squeeze()
			calcium.append(c_μ)
			spec.append(spec_rec)
			yn_samples.append(yn_sample.squeeze(1).squeeze(1))
			yb_samples.append(yb_sample.squeeze(1).squeeze(1))
		calcium = torch.cat(calcium, dim=0).detach().cpu().numpy()
		spec = torch.cat(spec, dim=0).detach().cpu().numpy()
		yn_samples = torch.cat(yn_samples, dim=0).detach().cpu().numpy()
		yb_samples = torch.cat(yb_samples, dim=0).detach().cpu().numpy()
	return calcium, spec, yn_samples, yb_samples


def write_diff_jpeg(temp, mag_factor=3, fn='temp.jpeg'):
	rgbArray = np.zeros((128,128,3), dtype='int')
	# temp_max = np.max(np.abs(temp))
	temp_max = np.quantile(np.abs(temp), 0.9975)
	print("max diff:", temp_max)
	temp /= temp_max
	# # Green/magenta.
	rgbArray[..., 0] = (-temp).clip(0,1)* 255 # R
	rgbArray[..., 1] = temp.clip(0,1)*255 # G
	rgbArray[..., 2] = (-temp).clip(0,1)*255 # B
	rgbArray[rgbArray<0] = 0
	rgbArray[rgbArray>255] = 255
	temp_rgbArray = np.zeros((128*mag_factor,128*mag_factor,3))
	for i in range(3):
		temp_rgbArray[:,:,i] = zoom(rgbArray[:,:,i].astype(np.float), mag_factor, order=3)
	rgbArray = temp_rgbArray.clip(0,255).astype(np.int)
	img = Image.fromarray(np.array(rgbArray, 'uint8'))
	img.save(fn)


def cca_scatters(model, nb_fn, n_neurons, model_cca=False):
	""" """
	assert not model_cca or model.model_type == 'poe_finch'
	# Get a model and a dataset.
	loaders = get_simple_test_train_loader(nb_fn=nb_fn)

	# Get local latents.
	train_latent = model.get_local_latents(loaders['train'])
	train_calcium, train_spec = train_latent['calcium'], train_latent['spec']
	test_latent = model.get_local_latents(loaders['test'])
	test_calcium, test_spec = test_latent['calcium'], test_latent['spec']
	print("n train:", len(train_spec))
	print("n test:", len(test_spec))

	# Generate fake data.
	if model.model_type == 'poe_finch':
		calcium, spec, yn_samples, yb_samples = generate_data(model, n_batches=80)
	else:
		calcium, spec, yn_samples, yb_samples, yn_proj, yb_proj = [None]*6

	# Reduce dimensionality with PCA.
	n_pca = PCA(n_components=32)
	b_pca = PCA(n_components=32)

	if model_cca:
		n_pca.fit(yn_samples)
		var_explained = np.cumsum(n_pca.explained_variance_ratio_)
		n_idx = np.searchsorted(var_explained, 0.5) + 1 # at least 99%
		print("calcium components kept", n_idx)
		b_pca.fit(yb_samples)
		var_explained = np.cumsum(b_pca.explained_variance_ratio_)
		b_idx = np.searchsorted(var_explained, 0.5) + 1 # at least 99%
		print("spec components kept", b_idx)
	else:
		n_pca.fit(train_calcium)
		var_explained = np.cumsum(n_pca.explained_variance_ratio_)
		n_idx = np.searchsorted(var_explained, 0.99) + 1 # at least 99%
		print("calcium components kept", n_idx)
		b_pca.fit(train_spec)
		var_explained = np.cumsum(b_pca.explained_variance_ratio_)
		b_idx = np.searchsorted(var_explained, 0.99) + 1 # at least 99%
		print("spec components kept", b_idx)
	# Fit samples.
	train_calcium = n_pca.transform(train_calcium)[:,:n_idx]
	test_calcium = n_pca.transform(test_calcium)[:,:n_idx]
	train_spec = b_pca.transform(train_spec)[:,:b_idx]
	test_spec = b_pca.transform(test_spec)[:,:b_idx]
	if model.model_type == 'poe_finch':
		yn_samples = n_pca.transform(yn_samples)[:,:n_idx]
		yb_samples = b_pca.transform(yb_samples)[:,:b_idx]

	# Perform CCA.
	n_cca_comp = min(5,b_idx,n_idx)
	cca = CCA(n_components=n_cca_comp, scale=False)
	if model_cca:
		cca.fit(yn_samples, yb_samples)
	else:
		cca.fit(train_calcium, train_spec)
	calcium_scores, spec_scores = cca.transform(train_calcium, train_spec)
	test_calcium_scores, test_spec_scores = cca.transform(test_calcium, test_spec)
	if model.model_type == 'poe_finch':
		yn_proj, yb_proj = cca.transform(yn_samples, yb_samples)

	train_corr, test_corr, null_corr_1, null_corr_2 = np.zeros(n_cca_comp), np.zeros(n_cca_comp), np.zeros(n_cca_comp), np.zeros(n_cca_comp)
	print()
	for i in range(calcium_scores.shape[1]):
		if model.model_type == 'poe_finch':
			model_corr = np.corrcoef(yn_proj[:,i], yb_proj[:,i])[0,1]
			print("CC", i+1, "model corr", model_corr)
		train_corr[i] = np.corrcoef(calcium_scores[:,i], spec_scores[:,i])[0,1]
		print("CC", i+1, "train corr", train_corr[i])
		test_corr[i] = np.corrcoef(test_calcium_scores[:,i], test_spec_scores[:,i])[0,1]
		print("CC", i+1, "test corr", test_corr[i])
	print()

	# Scatter.
	for cca_comp in range(n_cca_comp):
		plt.subplots(figsize=(3,3))
		all_c_scores = np.concatenate([calcium_scores[:,cca_comp], test_calcium_scores[:,cca_comp]], axis=0)[:]
		all_s_scores = np.concatenate([spec_scores[:,cca_comp], test_spec_scores[:,cca_comp]], axis=0)[:]
		# all_c_scores = (all_c_scores - np.mean(all_c_scores)) / np.std(all_c_scores)
		# all_s_scores = (all_s_scores - np.mean(all_s_scores)) / np.std(all_s_scores)
		reg = LinearRegression().fit(all_c_scores.reshape(-1,1), all_s_scores.reshape(-1,1))
		diff = np.diff(reg.predict([[0],[1]]).flatten())
		c_score_mean = np.mean(all_c_scores)
		temp_proj = (all_c_scores - c_score_mean)
		s_score_mean = np.mean(all_s_scores)
		temp_proj += (all_s_scores - s_score_mean)/diff
		temp_proj /= np.max(np.abs(temp_proj))
		temp_colors = np.zeros((len(temp_proj), 4))
		temp_colors[:,0] = (-temp_proj).clip(0,1)
		temp_colors[:,1] = temp_proj.clip(0,1)
		temp_colors[:,2] = (-temp_proj).clip(0,1)
		temp_colors[:,3] = 0.7
		plt.scatter(all_c_scores, all_s_scores, c=temp_colors)
		rho = np.corrcoef(test_calcium_scores[:,cca_comp], test_spec_scores[:,cca_comp])[0,1]
		plt.title(r'Test $\rho=$'+str(int(round(rho*100))/100))
		ax = plt.gca()
		for side in ['top', 'right']:
			ax.spines[side].set_visible(False)
		plt.xlabel("Neural CC "+str(cca_comp+1))
		plt.ylabel("Vocal CC "+str(cca_comp+1))
		plt.tight_layout()
		plt.savefig('cc'+str(cca_comp+1)+'.pdf')
		plt.close('all')
		# Estimate a p-value for the test correlation.
		rhos = np.zeros(100000)
		temp_calcium = test_calcium_scores[:,cca_comp]
		temp_spec = test_spec_scores[:,cca_comp]
		for j in range(rhos.shape[0]):
			perm = np.random.permutation(len(temp_calcium))
			rhos[j] = np.corrcoef(temp_calcium[perm], temp_spec)[0,1]
		rhos = rhos[np.argsort(rhos)]
		print("CC",cca_comp+1) # "quantiles:", np.quantile(rhos, [0.9,0.99,0.999,0.9999])
		null_corr_1[cca_comp] = np.quantile(rhos, 0.8)
		null_corr_2[cca_comp] = np.quantile(rhos, 0.95)
		# print("80% rho percentile: ", null_corr_1[cca_comp])
		# print("95% rho percentile: ", null_corr_2[cca_comp])
		print("p val estimate:", 1.0 - np.searchsorted(rhos, rho)/len(rhos))

		if model.model_type == 'poe_finch':
			# Plot the spectrogram vector:
			temp_proj = yn_proj[:,cca_comp] - c_score_mean
			temp_proj += (yb_proj[:,cca_comp] - s_score_mean)/diff
			temp_proj -= np.mean(temp_proj)
			diff_spec = np.mean(temp_proj.reshape(-1,1) * spec, axis=0).reshape(128,128)[::-1]
			diff_spec /= np.max(np.abs(diff_spec))
			write_diff_jpeg(diff_spec, fn='cc'+str(cca_comp+1)+'.jpeg')
			calcium_diff = np.mean(temp_proj.reshape(-1,1) * calcium, axis=0)
			savemat('cc'+str(cca_comp+1)+'.mat', {'diff': calcium_diff})


	# Plot the correlations.
	_ = plt.subplots(figsize=(2,2))
	x_vals = list(range(1,n_cca_comp+1))
	plt.scatter(x_vals, train_corr, c='r')
	plt.plot(x_vals, train_corr, c='k', alpha=0.4)
	plt.scatter(x_vals, test_corr, c='b')
	plt.plot(x_vals, test_corr, c='k', alpha=0.4)
	plt.fill_between(x_vals, np.zeros(len(x_vals)), null_corr_1, fc='k', alpha=0.2)
	plt.fill_between(x_vals, np.zeros(len(x_vals)), null_corr_2, fc='k', alpha=0.2)
	plt.xticks(x_vals)
	plt.savefig('temp.pdf')



if __name__ == '__main__':
	# model_fn = 'JUVI1_0305_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_20180305_spec_calcium.npy')
	# N_NEURONS = 76

	# model_fn = 'blk215_1005_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_10052018_spec_calcium.npy')
	# N_NEURONS = 70

	# model_fn = 'blk202_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk202/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
	# N_NEURONS = 44

	# model_fn = 'org137_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/org137/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
	# N_NEURONS = 78

	# model_fn = 'yel141_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/yel141/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
	# N_NEURONS = 36

	model_fn = 'juvi1_0228_vis_model.tar'
	DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
	NB_FN = os.path.join(DATA_DIR, 'nb_20180228_spec_calcium.npy')
	N_NEURONS = 41

	# model_fn = 'blk215_0929_vis_model.tar'
	# model_fn = 'blk215_0929_separate_vis_model.tar'
	# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
	# NB_FN = os.path.join(DATA_DIR, 'nb_09292018_spec_calcium.npy')
	# N_NEURONS = 44

	model = PoeFinch(model_type='poe_finch', spec_std=0.03, \
			calcium_std=0.3, fn=model_fn, calcium_dim=N_NEURONS)
	# model = CCAFinch(model_type='cca_finch', spec_std=0.03, \
	# 		calcium_std=0.3, fn=model_fn, calcium_dim=N_NEURONS)
	model.load_state()
	cca_scatters(model, NB_FN, N_NEURONS)
	quit()


###
