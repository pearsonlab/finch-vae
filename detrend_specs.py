"""
Try to correct for time in spectrograms and calcium.

"""

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from PIL import Image
from scipy.ndimage import zoom


IN_DATA_FN = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/nb_09292018_data.npy'
OUT_DATA_FN = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/nb_20180228_spec_calcium.npy'
MODE = 'CALCIUM_CORRECT'
assert MODE in ['SPEC_CORRECT', 'CALCIUM_CORRECT']

ALPHA_VALS = np.geomspace(1e-2, 1e2, 15)
GAMMA_VALS = np.geomspace(1e-6, 1e3, 15)



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


def get_times(data):
	"""Extract motif times from the .npy file, z-score, return."""
	fns = data['fn']
	onsets = data['onset']
	times = np.zeros(len(fns))
	for i in range(len(fns)):
		hour = float(fns[i][-10:-8])
		minute = float(fns[i][-8:-6])
		second = float(fns[i][-6:-4]) + onsets[i]
		times[i] = hour + minute/60 + second/3600
	time_mean = np.mean(times)
	times -= time_mean
	time_std = np.std(times)
	times /= time_std
	return times, time_mean, time_std


def choose_best_gamma(X, Y, n_splits=7):
	average_r2s = np.zeros(len(GAMMA_VALS)*len(ALPHA_VALS))
	kf = KFold(n_splits=n_splits)
	for train_index, test_index in kf.split(X):
		mean_train_Y = np.mean(Y[train_index], axis=0, keepdims=True)
		for i, gamma in enumerate(GAMMA_VALS):
			for j, alpha in enumerate(ALPHA_VALS):
				reg = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
				reg.fit(X[train_index], Y[train_index])
				pred_Y = reg.predict(X[test_index])
				term_1 = np.sum(np.power(Y[test_index] - mean_train_Y, 2))
				term_2 = np.sum(np.power(Y[test_index] - pred_Y, 2))
				r2 = 1.0 - term_2 / term_1
				average_r2s[i*len(ALPHA_VALS)+j] += r2
	idx = np.argmax(average_r2s)
	gamma = GAMMA_VALS[idx//len(ALPHA_VALS)]
	alpha = ALPHA_VALS[idx%len(ALPHA_VALS)]
	print("idx", idx, "gamma", gamma, "alpha", alpha)
	return gamma, alpha, average_r2s[idx]/n_splits



def get_variance(spec):
	return np.power(spec - np.mean(spec, axis=0, keepdims=True), 2).sum()



if __name__ == '__main__':
	data = np.load(IN_DATA_FN, allow_pickle=True).item()

	if MODE == 'SPEC_CORRECT':
		time, time_mean, time_std = get_times(data)
		time = time.reshape(-1,1)
		LASSO_REG = 1e-2

		print(data['spec'].shape)
		spec = data['spec'].reshape(time.shape[0],-1)
		orig_variance = get_variance(spec)

		# Correct for this dimension.
		itr = 0
		while True:
			itr += 1

			variance = get_variance(spec)
			print("Variance portion:", variance/orig_variance)

			reg = Lasso(alpha=LASSO_REG, fit_intercept=True, random_state=42).fit(time, spec)
			spec_vector = np.diff(reg.predict([[0.0],[1.0]]), axis=0)
			l2 = np.sqrt(np.sum(np.power(spec_vector, 2)))
			if l2 == 0.0:
				print("found nothing")
				break
			spec_vector /= l2
			write_diff_jpeg(np.copy(spec_vector).reshape(128,128)[::-1])
			# plt.imshow(spec_vector.reshape(128,128), origin='lower')
			# plt.colorbar()
			# plt.savefig('temp.pdf')
			# plt.close('all')
			_ = input("just plotted ")

			_, ax = plt.subplots(figsize=(4,2.7))
			for side in ['top', 'right']:
				ax.spines[side].set_visible(False)
			dot_products = np.sum(spec * spec_vector, axis=1)
			temp_time = (time * time_std + time_mean).flatten()
			plt.scatter(temp_time, dot_products.flatten(), c='b', alpha=0.5)
			mean_dot = np.mean(dot_products)
			# plt.axhline(y=mean_dot)
			gamma, alpha, avg_r2 = choose_best_gamma(time, dot_products.reshape(-1,1))
			reg = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma).fit(time, dot_products.reshape(-1,1))

			print("avg k-fold r2 =", avg_r2)
			if avg_r2 < 0.0:
				break
			x_vals = np.linspace(np.min(time), np.max(time), 200).reshape(-1,1)
			y_vals = reg.predict(x_vals)
			x_vals = x_vals * time_std + time_mean
			temp_plot = plt.plot(x_vals.flatten(), y_vals.flatten(), c='firebrick')[0]
			pred_dot_prod = reg.predict(time).flatten()
			r2 = 1.0 - np.sum(np.power(pred_dot_prod - dot_products.flatten(),2)) / np.sum(np.power(mean_dot - dot_products.flatten(),2))
			print("r2 =", r2)
			plt.xticks([10.25,10.5,10.75], ['10:15', '10:30', '10:45'])
			plt.ylabel('Spectrogram Projection (A.U.)')
			plt.xlabel('Time in Day')
			plt.tight_layout()
			plt.savefig('temp.pdf')
			plt.close('all')
			temp = input("continue? ")
			if temp == 'q':
				break
			for i in range(len(pred_dot_prod)):
				spec[i] += (mean_dot - pred_dot_prod[i]) * spec_vector.flatten()
			dot_products = np.sum(spec * spec_vector, axis=1)
			# plt.scatter(time.flatten(), dot_products.flatten(), c='r', alpha=0.5)

		# Save detrended specs.
		spec = spec.reshape(-1,128,128)
		data['spec'] = spec
		np.save(OUT_DATA_FN, data)
		quit()

	if MODE == 'CALCIUM_CORRECT':
		time, time_mean, time_std = get_times(data)
		print("min time", np.min(time * time_std + time_mean), "max time", np.max(time * time_std + time_mean))
		time = time.reshape(-1,1)
		LASSO_REG = 1e-2

		print(data['calcium'].shape)
		calcium = data['calcium'].reshape(time.shape[0],-1)
		orig_variance = get_variance(calcium)

		# Correct for this dimension.
		itr = 0
		while True:
			itr += 1

			variance = get_variance(calcium)
			print("Variance portion:", variance/orig_variance)

			reg = Lasso(alpha=LASSO_REG, fit_intercept=True, random_state=42).fit(time, calcium)
			calcium_vector = np.diff(reg.predict([[0.0],[1.0]]), axis=0)
			l2 = np.sqrt(np.sum(np.power(calcium_vector, 2)))
			if l2 == 0.0:
				print("found nothing")
				break
			calcium_vector /= l2
			plt.imshow(calcium_vector.reshape(1,-1), aspect='auto')
			plt.colorbar()
			plt.savefig('temp.pdf')
			plt.close('all')
			_ = input("just plotted ")

			_, ax = plt.subplots(figsize=(4,2.7))
			for side in ['top', 'right']:
				ax.spines[side].set_visible(False)
			dot_products = np.sum(calcium * calcium_vector, axis=1)
			temp_time = (time * time_std + time_mean).flatten()
			plt.scatter(temp_time, dot_products.flatten(), c='b', alpha=0.5)
			mean_dot = np.mean(dot_products)
			# plt.axhline(y=mean_dot)
			gamma, alpha, avg_r2 = choose_best_gamma(time, dot_products.reshape(-1,1))
			reg = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma).fit(time, dot_products.reshape(-1,1))

			print("avg k-fold r2 =", avg_r2)
			if avg_r2 < 0.0:
				break
			x_vals = np.linspace(np.min(time), np.max(time), 200).reshape(-1,1)
			y_vals = reg.predict(x_vals)
			x_vals = x_vals * time_std + time_mean
			print("min time", np.min(x_vals), "max time", np.max(x_vals))
			temp_plot = plt.plot(x_vals.flatten(), y_vals.flatten(), c='firebrick')[0]
			pred_dot_prod = reg.predict(time).flatten()

			r2 = 1.0 - np.sum(np.power(pred_dot_prod - dot_products.flatten(),2)) / np.sum(np.power(mean_dot - dot_products.flatten(),2))
			print("r2 =", r2)
			plt.xticks([10.25,10.5,10.75], ['10:15', '10:30', '10:45'])
			plt.ylabel('Calcium Projection (A.U.)')
			plt.xlabel('Time in Day')
			plt.tight_layout()
			plt.savefig('temp.pdf')
			plt.close('all')
			temp = input("continue? ")
			if temp == 'q':
				break
			for i in range(len(pred_dot_prod)):
				calcium[i] += (mean_dot - pred_dot_prod[i]) * calcium_vector.flatten()
			dot_products = np.sum(calcium * calcium_vector, axis=1)

		# Save detrended specs.
		data['calcium'] = calcium
		np.save(OUT_DATA_FN, data)
		quit()



###
