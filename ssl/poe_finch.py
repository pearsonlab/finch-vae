"""
Variational approximation to probabilistic CCA model with partially observed
data and an arbitrary number of views. Product of Gaussian experts approach.

Recognition models predict z-space directly.

Applying this to juvi1 dataset.

"""
__date__ = "May-August 2020"

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal, kl_divergence


MODEL_TYPES = ['poe_finch']
SPEC_DIM = 128*128
CALCIUM_DIM = 262
EPSILON = 1e-4

torch.autograd.set_detect_anomaly(True)


class PoeFinch(nn.Module):

	def __init__(self, z_dim=32, model_type='poe_finch', \
		fn='poe_finch_net.tar', device='auto', lr=8e-4, \
		spec_std=0.1, calcium_std='auto', calcium_dim=CALCIUM_DIM):
		"""
		Deep Variational CCA for Finch data.

		Parameters
		----------
		z_dim : int, optional
			Latent dimension.
		model_type : {...}, optional
			Which model to train.
		fn : str, optional
			Checkpoint filename.
		"""
		super(PoeFinch, self).__init__()
		self.model_type = model_type
		assert model_type in MODEL_TYPES
		self.z_dim = z_dim
		self.z_dim = z_dim # NOTE: hard-coded
		self.fn = fn
		self.spec_std = spec_std
		self.calcium_dim = calcium_dim
		if device == 'auto':
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		else:
			self.device = device
		# Spectrogram encoder
		self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
		self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
		self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
		self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
		self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
		self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
		self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
		self.bn1 = nn.BatchNorm2d(1)
		self.bn2 = nn.BatchNorm2d(8)
		self.bn3 = nn.BatchNorm2d(8)
		self.bn4 = nn.BatchNorm2d(16)
		self.bn5 = nn.BatchNorm2d(16)
		self.bn6 = nn.BatchNorm2d(24)
		self.bn7 = nn.BatchNorm2d(24)
		self.fc1 = nn.Linear(8192,1024)
		self.fc2 = nn.Linear(1024,256)
		self.fc31 = nn.Linear(256,64)
		self.fc32 = nn.Linear(256,64)
		self.fc33 = nn.Linear(256,64)
		self.fc41 = nn.Linear(64,self.z_dim) # mu
		self.fc42 = nn.Linear(64,self.z_dim) # d
		nn.init.normal_(self.fc33.weight, 0, 1e-6)
		nn.init.normal_(self.fc33.bias, 0, 1e-6)
		nn.init.normal_(self.fc42.weight, 0, 1e-6)
		nn.init.normal_(self.fc42.bias, -4, 1e-3)
		# Spectrogram decoder
		self.fc5 = nn.Linear(self.z_dim,64)
		self.fc6 = nn.Linear(64,256)
		self.fc7 = nn.Linear(256,1024)
		self.fc8 = nn.Linear(1024,8192)
		self.convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1)
		self.convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
		self.convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
		self.convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
		self.convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
		self.convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
		self.convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.bn9 = nn.BatchNorm2d(24)
		self.bn10 = nn.BatchNorm2d(24)
		self.bn11 = nn.BatchNorm2d(16)
		self.bn12 = nn.BatchNorm2d(16)
		self.bn13 = nn.BatchNorm2d(8)
		self.bn14 = nn.BatchNorm2d(8)
		# Neural recognition model.
		self.qn_1 = nn.Linear(self.calcium_dim, 128)
		self.qn_2 = nn.Linear(128,64)
		self.qn_31 = nn.Linear(64, self.z_dim) # mean
		self.qn_32 = nn.Linear(64, self.z_dim) # log stds
		# Neural model
		self.pn_1 = nn.Linear(self.z_dim, 64)
		self.pn_2 = nn.Linear(64,128)
		self.pn_3 = nn.Linear(128, self.calcium_dim)

		self.calcium_std = calcium_std
		if calcium_std == 'auto':
			self.calcium_log_std = nn.Parameter(-torch.ones(1,1,1,1))
		else:
			self.calcium_log_std = np.log(calcium_std)
		# Ws
		self.spec_W = nn.Parameter(torch.randn(1,1,1,self.z_dim,self.z_dim))
		self.calcium_W = nn.Parameter(torch.randn(1,1,1,self.z_dim,self.z_dim))
		# CCA noise
		self.y_log_std = nn.Parameter(torch.tensor(-1.0))
		# Optimizer
		self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5
		self.to(self.device)


	def save_state(self, fn=None):
		"""Save state."""
		if fn is None:
			fn = self.fn
		torch.save({
				'model_state_dict': self.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
			}, fn)


	def load_state(self, fn=None):
		"""Load state."""
		if fn is None:
			fn = self.fn
		print("Loading state from:", fn)
		checkpoint = torch.load(fn, map_location=self.device)
		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


	def encode_spec(self, spec, n_samples=1):
		""" """
		h = spec.unsqueeze(-1).view(-1,1,1,128,128).squeeze(-3)
		h = F.relu(self.conv1(self.bn1(h)))
		h = F.relu(self.conv2(self.bn2(h)))
		h = F.relu(self.conv3(self.bn3(h)))
		h = F.relu(self.conv4(self.bn4(h)))
		h = F.relu(self.conv5(self.bn5(h)))
		h = F.relu(self.conv6(self.bn6(h)))
		h = F.relu(self.conv7(self.bn7(h)))
		h = h.view(-1, 1, 8192).unsqueeze(-2)
		h = F.relu(self.fc1(h))
		h = F.relu(self.fc2(h))
		mu = F.relu(self.fc31(h))
		mu = self.fc41(mu)
		log_d = F.relu(self.fc32(h))
		log_d = self.fc42(log_d)
		return mu.expand(-1,n_samples,-1,-1), log_d.exp().expand(-1,n_samples,-1,-1)


	def encode_calcium(self, calcium, n_samples=1):
		h = F.relu(self.qn_1(calcium))
		h = F.relu(self.qn_2(h))
		n_μ = self.qn_31(h)
		n_log_d = self.qn_32(h)
		return n_μ.expand(-1,n_samples,-1,-1), n_log_d.exp().expand(-1,n_samples,-1,-1)


	def decode_spec(self, y):
		""" """
		h = F.relu(self.fc5(y))
		h = F.relu(self.fc6(h))
		h = F.relu(self.fc7(h))
		h = F.relu(self.fc8(h))
		h = h.view(-1,32,16,16)
		h = F.relu(self.convt1(self.bn8(h)))
		h = F.relu(self.convt2(self.bn9(h)))
		h = F.relu(self.convt3(self.bn10(h)))
		h = F.relu(self.convt4(self.bn11(h)))
		h = F.relu(self.convt5(self.bn12(h)))
		h = F.relu(self.convt6(self.bn13(h)))
		h = self.convt7(self.bn14(h))
		return h.view(y.shape[:-1]+(SPEC_DIM,))


	def decode_calcium(self, y):
		h = F.relu(self.pn_1(y))
		h = F.relu(self.pn_2(h))
		return self.pn_3(h)


	def multiply_experts(self, μs, precs, batch, n_samples, return_kl=True):
		""" """
		# Add the prior expert.
		μs.append(torch.zeros(batch, n_samples, 1, self.z_dim, device=self.device))
		precs.append(torch.ones(batch, n_samples, 1, self.z_dim, device=self.device))
		# Multiply the Gaussians.
		prec = torch.cat(precs, dim=-2).sum(-2, keepdim=True)
		prec_μ = (prec * torch.cat(μs, dim=-2).sum(-2, keepdim=True)).sum(-2, keepdim=True)
		q_Σ = torch.reciprocal(prec)
		q_std = q_Σ.sqrt()
		q_mean = q_Σ * prec_μ
		if return_kl:
			kl_z = 0.5 * (q_Σ + q_mean.pow(2) - q_Σ.log() - 1.0).sum(dim=-1, keepdim=True)
			return q_mean, q_std, kl_z
		return q_mean, q_std


	def sample_z(self, q_mean, q_std):
		""" """
		q_dist = Normal(q_mean, q_std+EPSILON)
		p_dist = Normal(torch.zeros_like(q_mean), torch.ones_like(q_std))
		z_sample = q_dist.rsample()
		log_qz = q_dist.log_prob(z_sample).sum(-1, keepdim=True)
		log_pz = p_dist.log_prob(z_sample).sum(-1, keepdim=True)
		return z_sample, log_qz, log_pz


	def sample_yb(self, z_sample):
		""" """
		# Multiply and add noise.
		b_y_sample = (self.spec_W @ z_sample.unsqueeze(-1)).squeeze(-1)
		noise = self.y_log_std * torch.randn(b_y_sample.shape, device=self.device)
		return b_y_sample + noise


	def sample_yn(self, z_sample):
		""" """
		# Multiply and add noise.
		n_y_sample = (self.calcium_W @ z_sample.unsqueeze(-1)).squeeze(-1)
		noise = self.y_log_std * torch.randn(n_y_sample.shape, device=self.device)
		return n_y_sample + noise


	def _format_input(self, neurons, spec, n_samples):
		"""
		Make everything shape: [batch,n_samples,n_views,view_dim]

		neurons : [batch,n_views]
		spec : [batch,spec_dim]
		"""
		neurons.unsqueeze_(-2).unsqueeze_(-2).expand(-1,n_samples,-1,-1)
		spec.unsqueeze_(-2).unsqueeze_(-2).expand(-1,n_samples,-1,-1)


	def forward(self, calcium, spec, β_z=1.0, only_encode=False, \
		return_rec=False, n_samples=1, iwae_elbo=False, λ_nb=1.0, λ_n=1.0, λ_b=1.0):
		"""
		Encode, decode, and return loss.

		* Almost everything has shape [batch,n_samples,n_views,dim]
		"""
		assert n_samples == 1 or iwae_elbo
		self.train()
		self._format_input(calcium, spec, n_samples)
		batch = calcium.shape[-4]
		# Mask neural data.
		calcium_mask = torch.isnan(calcium)
		calcium[calcium_mask] = 0.0
		# Encode neural data.
		n_μ, n_prec = self.encode_calcium(calcium, n_samples=n_samples)
		# Encode spectrograms.
		b_μ, b_prec = self.encode_spec(spec, n_samples=n_samples)
		# Product of experts: all 3 subsets.
		q_mean_n, q_std_n, kl_z_n = self.multiply_experts([n_μ], [n_prec], batch, n_samples)
		q_mean_b, q_std_b, kl_z_b = self.multiply_experts([b_μ], [b_prec], batch, n_samples)
		q_mean_nb, q_std_nb, kl_z_nb = self.multiply_experts([n_μ, b_μ], [n_prec, b_prec], batch, n_samples)
		if only_encode: # return just the variational posteriors
			return q_mean_nb, q_std_nb
		# Sample latent zs.
		z_sample_n, log_qz_n, log_pz_n = self.sample_z(q_mean_n, q_std_n)
		z_sample_b, log_qz_b, log_pz_b = self.sample_z(q_mean_b, q_std_b)
		z_sample_nb, log_qz_nb, log_pz_nb = self.sample_z(q_mean_nb, q_std_nb)
		# Sample latent ys.
		yn_sample_n = self.sample_yn(z_sample_n)
		yb_sample_b = self.sample_yb(z_sample_b)
		yn_sample_nb = self.sample_yn(z_sample_nb)
		yb_sample_nb = self.sample_yb(z_sample_nb)
		# Reconstruct calcium.
		if self.calcium_std == 'auto':
			calcium_dist = Normal(calcium, self.calcium_log_std.exp()+EPSILON)
		else:
			calcium_dist = Normal(calcium, np.exp(self.calcium_log_std))
		c_μ_nb = self.decode_calcium(yn_sample_nb)
		c_μ_n = self.decode_calcium(yn_sample_n)
		calcium_logp_nb = calcium_dist.log_prob(c_μ_nb)
		calcium_logp_n = calcium_dist.log_prob(c_μ_n)
		calcium_logp_nb[calcium_mask.expand(-1,n_samples,-1,-1)] = 0.0
		calcium_logp_n[calcium_mask.expand(-1,n_samples,-1,-1)] = 0.0
		calcium_logp_nb = calcium_logp_nb.sum((-1,-2), keepdim=True)
		calcium_logp_n = calcium_logp_n.sum((-1,-2), keepdim=True)
		# Reconstruct spectrogram.
		spec_dist = Normal(spec, self.spec_std)
		spec_rec_nb = self.decode_spec(yb_sample_nb)
		spec_rec_b = self.decode_spec(yb_sample_b)
		spec_logp_nb = spec_dist.log_prob(spec_rec_nb).sum((-1,-2), keepdim=True)
		spec_logp_b = spec_dist.log_prob(spec_rec_b).sum((-1,-2), keepdim=True)
		# Estimate an evidence lower bound (ELBO).
		if iwae_elbo: # importance-weighted ELBO
			elbo_nb = log_pz_nb + calcium_logp_nb + spec_logp_nb - log_qz_nb - np.log(n_samples)
			elbo_nb = torch.logsumexp(elbo_nb, dim=1).sum()
			elbo_n = log_pz_n + calcium_logp_n - log_qz_n - np.log(n_samples)
			elbo_n = torch.logsumexp(elbo_n, dim=1).sum()
			elbo_b = log_pz_b + spec_logp_b - log_qz_b - np.log(n_samples)
			elbo_b = torch.logsumexp(elbo_b, dim=1).sum()
		else: # standard single-sample ELBO
			elbo_nb = (spec_logp_nb + calcium_logp_nb - β_z*kl_z_nb).sum()
			elbo_n = (calcium_logp_n - β_z*kl_z_n).sum()
			elbo_b = (spec_logp_b - β_z*kl_z_b).sum()
		# Weight the ELBOs.
		loss = -(λ_nb*elbo_nb + λ_n*elbo_n + λ_b*elbo_b) / batch
		# Return.
		if return_rec:
			return loss, calcium_logp_nb.sum(), c_μ_nb.detach().cpu().numpy(), spec_rec_nb.detach().cpu().numpy()
		return loss, calcium_logp_nb.sum()


	def get_local_latents(self, loader):
		""" """
		calcium = loader.dataset.neural.to(self.device)
		spec = loader.dataset.behavioral.to(self.device)
		times = loader.dataset.times
		with torch.no_grad():
			self._format_input(calcium, spec, 1)
			calcium_mask = torch.isnan(calcium)
			calcium[calcium_mask] = 0.0
			n_μ, _ = self.encode_calcium(calcium, n_samples=1)
			n_μ = (self.calcium_W @ n_μ.unsqueeze(-1)).squeeze(-1)
			b_μ, _ = self.encode_spec(spec, n_samples=1)
			b_μ = (self.spec_W @ b_μ.unsqueeze(-1)).squeeze(-1)
			n_μ.squeeze_(-2).squeeze_(-2)
			b_μ.squeeze_(-2).squeeze_(-2)
		return { \
			'calcium': n_μ.detach().cpu().numpy(),
			'spec': b_μ.detach().cpu().numpy(),
			'times': times,
			'tempo': loader.dataset.tempo,
		}


	def rec_plot(self, loader, save=False):
		"""
		Plot reconstructed spectrograms and calcium levels.
		"""
		for batch in loader:
			n = batch['n'][:1].to(self.device)
			b = batch['b'][:1].to(self.device)
			with torch.no_grad():
				_, _, n_rec, b_rec = self.forward(n, b, return_rec=True)
				n = n.detach().cpu().numpy().flatten()
				b = b.detach().cpu().numpy().reshape(128,128)
				n_rec = n_rec[0,0].flatten()
				b_rec = b_rec[0,0].reshape(128,128)
				break
		if save:
			d = {'neural': n, 'neural_rec': n_rec, 'vocal': b, 'vocal_rec': b_rec}
			np.save('rec_data.npy', d)
		fig = plt.figure(constrained_layout=True, figsize=(5,4))
		gs = GridSpec(2, 2, figure=fig, height_ratios=[0.4,0.6])
		ax0 = fig.add_subplot(gs[0,:])
		ax1 = fig.add_subplot(gs[1,0])
		ax2 = fig.add_subplot(gs[1,1])
		perm = np.argsort(n_rec)
		x_vals = np.arange(len(n_rec))
		ax0.bar(x_vals-0.15,n[perm],align='center',color='tomato',width=0.3)
		ax0.bar(x_vals+0.15,n_rec[perm],align='center',color='deepskyblue',width=0.3)
		ax0.spines['right'].set_visible(False)
		ax0.spines['top'].set_visible(False)
		ax0.spines['bottom'].set_visible(False)
		ax0.set_xticks([])
		ax0.set_xticklabels([])
		ax0.set_ylabel('calcium signal')
		ax0.set_xlabel('units')
		ax1.imshow(b, origin='lower', vmin=0, vmax=1)
		ax1.axis('off')
		ax1.set_title('Real', color='tomato')
		ax2.imshow(b_rec, origin='lower', vmin=0, vmax=1)
		ax2.axis('off')
		ax2.set_title('Reconstruction', color='deepskyblue')
		plt.savefig('temp.pdf')
		plt.close('all')



	def train_epoch(self, loader, optimize=True):
		"""
		Train for a single epoch.

		Parameters
		----------
		loader : torch.utils.data.DataLoader
			DataLoader

		Returns
		-------
		epoch_loss : float
			Negative ELBO estimate.
		"""
		self.train()
		epoch_loss = 0.0
		epoch_mses = []
		for i, batch in enumerate(loader):
			self.optimizer.zero_grad()
			n = batch['n'].to(self.device)
			b = batch['b'].to(self.device)
			loss, mse = self.forward(n, b)
			epoch_loss += loss.item() * len(batch['n'])
			epoch_mses.append(mse.item())
			if optimize:
				loss.backward()
				self.optimizer.step()
		return epoch_loss / len(loader.dataset), sum(epoch_mses) / len(epoch_mses)



if __name__ == '__main__':
	pass



###
