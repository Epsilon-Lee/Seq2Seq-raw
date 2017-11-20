import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Generator(nn.Module):
	"""A feed forward neural network which takes decoder'output
	and produce estimated distribution over vocabulary. 
	"""
	def __init__(
		self,
		vocab_size,
		dec_hid_size,
		dim_lst,
		num_layers
	):
		self.vocab_size = vocab_size
		self.dec_hid_size = dec_hid_size
		self.dim_lst = dim_lst
		self.num_layers = num_layers

		if self.num_layers != len(self.dim_lst):
			assert False, "Generator's layer number does not match dim_lst"

		if self.dec_hid_size != dim_lst[0]:
			assert False, "First layer's input dim does not match decoder's output dim"

		super(Generator, self).__init__()

		seq_module_lst = []
		for l in range(self.num_layers - 1):
			seq_module_lst.append(
				(
					str(l + 1) + '-linear',
					nn.Linear(
						self.dim_lst[l],
						self.dim_lst[l + 1]
					)
				)
			)
			seq_module_lst.append(
				(
					str(l + 1) + '-relu',
					nn.ReLU()
				)
			)
		seq_module_lst.append(
			(
				str(self.num_layers) + '-linear',
				nn.Linear(
					self.dim_lst[-1],
					self.vocab_size
				)
			)
		)
		# seq_module_lst.append(
		# 	(
		# 		str(self.num_layers) + '-softmax',
		# 		nn.Softmax()
		# 	)
		# )

		self.ff_networks = nn.Sequential(
			OrderedDict(seq_module_lst)
		)

	def forward(self, dec_hids):
		"""Forward compute method for Generator

		Args
		----------
		dec_hids   : N x L x H

		Return
		----------
		voc_dstrs  : N x L x vocab_size
			This is the distributions over vocabulary at each time step
			for each example, and the score is normalized through a Softmax
			layer. 

		"""
		# dec_hids_2d = dec_hids.view(-1, dec_hids.size(2))
		# return F.log_softmax(self.ff_networks(dec_hids))
		return F.softmax(self.ff_networks(dec_hids))