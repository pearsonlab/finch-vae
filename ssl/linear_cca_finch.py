"""
Separate encoding model with and without post-hoc CCA, all linear models.


"""
__date__ = "May - Octoboer 2020"

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal, kl_divergence


MODEL_TYPES = ['linear_cca_finch']
SPEC_DIM = 128*128
CALCIUM_DIM = 262
EPSILON = 1e-4

torch.autograd.set_detect_anomaly(True)


class CCAFinch(nn.Module):

	def __init__(self, z_dim=32, model_type='linear_cca_finch', \
		fn='cca_finch_net.tar', device='auto', lr=8e-4, spec_std=0.1, \
		calcium_std='auto', calcium_dim=CALCIUM_DIM):
		"""
		Two VAEs with a post-hoc CCA model for Finch data.

		Parameters
		----------
		z_dim : int, optional
			Latent dimension.
		model_type : {...}, optional
			Which model to train.
		fn : str, optional
			Checkpoint filename.
		"""
		super(CCAFinch, self).__init__()
		self.model_type = model_type
		assert model_type in MODEL_TYPES
		self.z_dim = z_dim
		self.spec_std = spec_std # NOTE: also hard-coded
		self.fn = fn
		self.calcium_dim = calcium_dim
		if device == 'auto':
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		else:
			self.device = device
		# Spectrogram layers
		self.qb_1 = nn.Linear(SPEC_DIM, self.z_dim)
		self.qb_2 = nn.Linear(SPEC_DIM, self.z_dim)
		self.pb_1 = nn.Linear(self.z_dim, SPEC_DIM)
		# Calcium layers
		self.qn_1 = nn.Linear(self.calcium_dim, self.z_dim)
		self.qn_2 = nn.Linear(self.calcium_dim, self.z_dim)
		self.pn_1 = nn.Linear(self.z_dim, self.calcium_dim)
		self.calcium_std = calcium_std
		if calcium_std == 'auto':
			self.calcium_log_std = nn.Parameter(-torch.ones(1,1,1,1))
		else:
			self.calcium_log_std = np.log(calcium_std)
		# Optimizer
		self.optimizer = Adam(self.parameters(), lr=lr) # 8e-5
		# print("num parameters:", sum(p.numel() for p in self.parameters()))
		# quit()
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
		mu = self.qb_1(spec)
		log_d = self.qb_2(spec)
		return mu.expand(-1,n_samples,-1,-1), log_d.exp().expand(-1,n_samples,-1,-1)


	def encode_calcium(self, calcium, n_samples=1):
		""" """
		n_μ = self.qn_1(calcium)
		n_log_d = self.qn_2(calcium)
		return n_μ.expand(-1,n_samples,-1,-1), n_log_d.exp().expand(-1,n_samples,-1,-1)


	def decode_spec(self, y):
		""" """
		return self.pb_1(y)


	def decode_calcium(self, y):
		""" """
		return self.pn_1(y)


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


	def sample_y(self, q_mean, q_std):
		""" """
		q_dist = Normal(q_mean, q_std+EPSILON)
		p_dist = Normal(torch.zeros_like(q_mean), torch.ones_like(q_std))
		z_sample = q_dist.rsample()
		log_qz = q_dist.log_prob(z_sample).sum(-1, keepdim=True)
		log_pz = p_dist.log_prob(z_sample).sum(-1, keepdim=True)
		return z_sample, log_qz, log_pz


	def _format_input(self, neurons, spec, n_samples):
		"""
		Make everything shape: [batch,n_samples,n_views,view_dim]

		neurons : [batch,n_views]
		spec : [batch,spec_dim]
		"""
		neurons.unsqueeze_(-2).unsqueeze_(-2).expand(-1,n_samples,-1,-1)
		spec.unsqueeze_(-2).unsqueeze_(-2).expand(-1,n_samples,-1,-1)


	def forward(self, calcium, spec, β_z=1.0, only_encode=False, \
		return_rec=False, n_samples=1, iwae_elbo=False):
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
		# Get separate variational posteriors.
		qn_mean, qn_std, kl_zn = self.multiply_experts([n_μ], [n_prec], batch, n_samples)
		qb_mean, qb_std, kl_zb = self.multiply_experts([b_μ], [b_prec], batch, n_samples)
		if only_encode:
			return qn_mean, qn_std, qb_mean, qb_std
		# Sample latent ys.
		yn_sample, log_qzn, log_pzn = self.sample_y(qn_mean, qn_std)
		yb_sample, log_qzb, log_pzb = self.sample_y(qb_mean, qb_std)
		# Reconstruct calcium.
		c_μ = self.decode_calcium(yn_sample)
		if self.calcium_std == 'auto':
			calcium_dist = Normal(c_μ, self.calcium_log_std.exp()+EPSILON)
		else:
			calcium_dist = Normal(c_μ, np.exp(self.calcium_log_std))
		calcium_logp = calcium_dist.log_prob(calcium)
		calcium_logp[calcium_mask.expand(-1,n_samples,-1,-1)] = 0.0
		calcium_logp = calcium_logp.sum((-1,-2), keepdim=True)
		# Reconstruct spectrogram.
		spec_rec = self.decode_spec(yb_sample)
		spec_dist = Normal(spec_rec, self.spec_std)
		spec_logp = spec_dist.log_prob(spec).sum((-1,-2), keepdim=True)
		# Estimate an evidence lower bound (ELBO).
		if iwae_elbo: # importance-weighted ELBO
			elbo_n = log_pzn + calcium_logp - log_qzn - np.log(n_samples)
			elbo_n = torch.logsumexp(elbo_n, dim=1).sum()
			elbo_b = log_pzb + spec_logp - log_qzb - np.log(n_samples)
			elbo_b = torch.logsumexp(elbo_b, dim=1).sum()
		else: # standard single-sample ELBO
			elbo_n = (calcium_logp - β_z*kl_zn).sum()
			elbo_b = (spec_logp - β_z*kl_zb).sum()
		loss = -(elbo_n + elbo_b) / batch
		# Return.
		if return_rec:
			return loss, calcium_logp.sum(), c_μ.detach().cpu().numpy(), spec_rec.detach().cpu().numpy()
		return loss, calcium_logp.sum()


	def get_local_latents(self, loader):
		""" """
		calcium = loader.dataset.neural.to(self.device)
		spec = loader.dataset.behavioral.to(self.device)
		times = loader.dataset.times
		with torch.no_grad():
			qn_mean, _, qb_mean, _ = self.forward(calcium, spec, n_samples=1, only_encode=True)
			qn_mean.squeeze_(-2).squeeze_(-2)
			qb_mean.squeeze_(-2).squeeze_(-2)
		return {
			'calcium': qn_mean.detach().cpu().numpy(),
			'spec': qb_mean.detach().cpu().numpy(),
			'times': times,
			'tempo': loader.dataset.tempo,
		}


	def rec_plot(self, loader):
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
