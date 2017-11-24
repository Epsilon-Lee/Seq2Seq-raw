import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BahdahnauAttention(nn.Module): 
	"""In the original paper's appendix, the decoder in their
	paper uses gated recurrent unit (GRU) as cell, and adds context
	vector c_i to the GRU as part of the information for GRU to
	consider. It is not possible to implement this with PyTorch's
	GRU interface. However we could concatenate the c_i with s_{i-1}
	and do a linear transform to make it have the same dimension of
	s_{i-1}, the linear projection plays a role in computation of s'_i:
		s_{i-1}_projected = V [s_{i-1}, c_i]
		s'_i = tanh(Wy_{i-1} + U[r_i s_{i-1}_projected])

	s_{i-1}            : the previous hidden state
	s_{i-1}_projected  : the linear projection of the previous hidden
	                     state concatenated with c_i

	alpha_ij_unnormalized = v tanh(W_a s_{i-1} + U_a h_j) # N x 1 x H + N x L x H => N x L x 1
	alpha_ij = F.softmax(alpha_ij_unnormalized) => N x L
	c_i = \sum_j alpha_ij * h_j => N x L x 1 * N x L x H => N x H

	"""
	def __init__(
		self,
		enc_hid_size,
		enc_num_dirs,
		dec_hid_size
	):
		self.enc_hid_size = enc_hid_size
		self.enc_num_dirs = enc_num_dirs
		self.dec_hid_size = dec_hid_size

		super(BahdahnauAttention, self).__init__()

		self.project_dec_hids = nn.Linear(
			self.dec_hid_size,
			self.dec_hid_size
		)

		self.project_enc_hids = nn.Linear(
			self.enc_hid_size * self.enc_num_dirs,
			self.dec_hid_size
		)

		self.project_to_one_dim = nn.Linear(
			self.dec_hid_size,
			1
		)

	def forward(self, s_prev, enc_hids):
		"""Forward compute method

		Args
		----------
		s_prev    : N x dec_H
		enc_hids  : N x enc_L x (enc_H x enc_nDir)

		Return
		----------
		c_curr    : N x enc_H
		att_curr  : N x enc_L
		"""
		s_prev_projected = self.project_dec_hids(s_prev).unsqueeze(1) # N x 1 x dec_H
		# print(type(enc_hids))
		# print(s_prev_projected.size())
		# print(enc_hids.size())
		enc_hids_projected = self.project_enc_hids(enc_hids) # N x L x dec_H
		alpha_unnormalized = self.project_to_one_dim(
			F.tanh(s_prev_projected + enc_hids_projected)
		).squeeze(2) # N x L
		alpha = F.softmax(alpha_unnormalized).unsqueeze(2) # N x L x 1
		# print(alpha.size())
		# print(enc_hids.size())
		enc_hids = enc_hids * alpha # N x L x enc_H
		c_curr = torch.sum(enc_hids, dim=1) # N x enc_H
		att_curr = alpha.squeeze(2)

		return c_curr, att_curr


"""The following two classes are implementations of the paper:

['Effective Approaches to Attention-based Neural Machine Translation']
(http://aclweb.org/anthology/D15-1166)

- Global attention : like Bahdahnau's attention mechanism however use 
                     current hidden output to compute the context vector

                     - Dot
                     - Bilinear
                     - Concate + MLP

- Local attention  : to predict a probability score which has its semantics
                     when multiply by the encoder side sentence length; this
                     means a position `p` to put attention over.

                     - Monotonic: [p - D, p + D] 2D + 1 words as context
                     - Gaussian : TODO

"""

class GlobalAttention(nn.Module):
	"""Global attention 

	"""
	def __init__(
		self,
		attention_type,
		enc_hid_size,
		enc_num_dirs,
		dec_hid_size
	):
		self.attention_type = attention_type
		self.enc_hid_size = enc_hid_size
		self.enc_num_dirs = enc_num_dirs
		self.dec_hid_size = dec_hid_size

		super(GlobalAttention, self).__init__()

		if self.attention_type == 'dot' and self.enc_hid_size * self.enc_num_dirs != self.dec_hid_size:
			assert False, "When dot attention, %d number of encoder dim should match decoder" % self.enc_num_dirs
			
		elif self.attention_type == 'bilinear':
			self.project_dec_hids = nn.Linear(
				self.dec_hid_size,
				self.enc_hid_size * sef.enc_num_dirs
			)
		else:
			self.project_dec_hids = nn.Linear(
				self.dec_hid_size,
				self.dec_hid_size
			)
			self.project_enc_hids = nn.Linear(
				self.enc_hid_size * self.enc_num_dirs,
				self.dec_hid_size
			)
			self.project_to_one_dim = nn.Linear(
				self.dec_hid_size,
				1
			)
		self.project_att_hids = nn.Linear(
			self.enc_hid_size * self.enc_num_dirs,
			self.dec_hid_size
		)

	def forward(self, dec_h_curr, enc_hids):
		"""Forward compute method

		Args
		----------
		dec_h_curr  : N x dec_H
		enc_hids    : N x L x (enc_H x enc_nDir)

		Return
		----------
		h_att_curr  : N x dec_H
		att_curr    : N x L

		"""
		if self.attention_type == 'dot':
			dec_h_curr = dec_h_curr.unsqueeze(2)
			alpha_unnormalized = torch.bmm(dec_hids, dec_h_curr) # N x (L x enc_H x dec_H x 1) => N x L x 1
			alpha = F.softmax(alpha_unnormalized) # N x L
			enc_hids = alpha.unsqueeze(2) * enc_hids # N x L x enc_H
			h_att_curr = torch.sum(enc_hids, dim=1) # N x enc_H
			h_att_curr = self.project_att_hids(h_att_curr)
			att_curr = alpha
			return h_att_curr, att_curr
		elif self.attention_type == 'bilinear':
			dec_h_curr_projected = self.project_dec_hids(dec_h_curr).unsqueeze(2) # N x enc_H x 1
			alpha_unnormalized = torch.bmm(
				dec_hids,
				dec_h_curr_projected
			).squeeze(2) # N x L
			alpha = F.softmax(alpha_unnormalized) # N x L
			enc_hids = alpha.unsqueeze(2) * enc_hids
			h_att_curr = torch.sum(enc_hids, dim=1)
			h_att_curr = self.project_att_hids(h_att_curr)
			att_curr = alpha
			return h_att_curr, att_curr
		else:
			dec_h_curr_projected = self.project_dec_hids(dec_h_curr) # N x dec_H
			enc_hids_projected = self.project_enc_hids(enc_hids) # N x L x dec_H
			alpha_unnormalized = self.project_to_one_dim(
				F.tanh(dec_h_curr_projected.unsqueeze(1) + enc_hids_projected)
			).squeeze(2) # N x L
			alpha = F.softmax(alpha_unnormalized) # N x L
			enc_hids = alpha.unsqueeze(2) * enc_hids
			h_att_curr = torch.sum(enc_hids, dim=1)
			h_att_curr = self.project_att_hids(h_att_curr)
			att_curr = alpha
			return h_att_curr, att_curr


class LocalAttention(nn.Module):
	"""Local attention

	"""
	def __init__(
		self,
		dec_hid_size,
		enc_hid_size
	):
		self.dec_hid_size = dec_hid_size
		self.enc_hid_size = enc_hid_size

		super(LocalAttention, self).__init__()

		self.position_prediction_net = nn.Sequential(
			nn.Linear(self.dec_hid_size, self.dec_hid_size / 2),
			nn.ReLU(),
			nn.Linear(self.dec_hid_size / 2, 1),
			nn.Sigmoid()
		)

	def forward(self, dec_h_curr, enc_hids, input_lengths):
		"""Forward compute method

		Args
		----------
		dec_h_curr    : N x dec_H
		enc_hids      : N x L x enc_H
		input_lengths : list len() => N

		Return
		----------
		h_att_curr    : N x dec_H
		att_curr      : N x L

		"""
		position_ratio = self.position_prediction_net(dec_h_curr) # N x 1
		lengths_var = Variable(torch.FloatTensor(input_lengths)).unsqueeze(1) # N x 1
		positions = lengths_var * position_ratio # N x 1