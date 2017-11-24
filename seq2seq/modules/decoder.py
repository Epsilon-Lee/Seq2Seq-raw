import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import BahdahnauAttention, GlobalAttention


class StackedRNNCell(nn.Module):
	"""Stacked RNN Cell, which is used for
	one-step computation forward.
	"""
	def __init__(
		self,
		num_layers,
		emb_size,
		rnn_cell_type,
		hid_size,
		dropout
	):
		self.num_layers = num_layers
		self.emb_size = emb_size
		self.rnn_cell_type = rnn_cell_type
		self.hid_size = hid_size
		self.dropout = dropout
		
		super(StackedRNNCell, self).__init__()

		self.dropout_layer = nn.Dropout(p=self.dropout)

		self.rnn_stack = nn.ModuleList()
		if self.rnn_cell_type == 'rnn':
			self.rnn_stack.append(
				nn.RNNCell(
					self.emb_size,
					self.hid_size
				)
			)
			for l in range(self.num_layers - 1):
				self.rnn_stack.append(
					nn.RNNCell(
						self.hid_size,
						self.hid_size
					)
				)
		elif self.rnn_cell_type == 'lstm':
			self.rnn_stack.append(
				nn.LSTMCell(
					self.emb_size,
					self.hid_sizes
				)
			)
			for l in range(self.num_layers - 1):
				self.rnn_stack.append(
					nn.LSTMCell(
						self.hid_size,
						self.hid_size
					)
				)
		else:
			self.rnn_stack.append(
				nn.GRUCell(
					self.emb_size,
					self.hid_size
				)
			)
			for l in range(self.num_layers - 1):
				self.rnn_stack.append(
					nn.GRUCell(
						self.hid_size,
						self.hid_size
					)
				)
	
	def forward(self, input, prev):
		"""Forward compute for each time-step

		Args
		----------
		input  : N x Embz torch.FloatTensor
		prev   : tuple (prev_h, prev_c) / nL x N x H

		Return
		----------
		output : N x H
		curr   : tuple (curr_h, curr_c)/ nL x N x H

		"""
		if type(prev) == tuple:
			prev_h, prev_c = prev
		else:
			prev_h = prev

		layer_input = input
		curr_h = []
		curr_c = []
		if type(prev) == tuple:
			# layer_fn     : RNNCell
			# layer_prev_h : 1 x N x H
			# layer_prev_c : 1 x N x H
			for layer_fn, layer_prev_h, layer_prev_c in zip(
				self.rnn_stack,
				prev_h.split(1),
				prev_c.split(1)
			):
				layer_prev_h = layer_prev_h.squeeze(0) # N x H
				layer_prev_c = layer_prev_c.squeeze(0)
				layer_input, layer_c  = layer_fn(
					layer_input, 
					(layer_prev_h, layer_prev_c)
				)
				layer_input = self.dropout_layer(layer_input) # dropout
				curr_h.append(layer_input)
				curr_c.append(layer_c)
		else:
			for layer_fn, layer_prev_h in zip(
				self.rnn_stack,
				prev_h.split(1)
			):
				layer_prev_h = layer_prev_h.squeeze(0)
				layer_input = layer_fn(
					layer_input,
					layer_prev_h
				)
				layer_input = self.dropout_layer(layer_input) # dropout
				curr_h.append(layer_input)

		curr_h = torch.stack(curr_h, dim=0) # nL x N x H
		if type(prev) == tuple:
			curr_c = curr_c.stack(dim=0)
			return layer_input, (curr_h, curr_c)
		else:
			return layer_input, curr_h


class NaiveDecoder(nn.Module):
	"""A decoder class a with naive un-attentive RNN architecture
	"""
	def __init__(
		self,
		decoder_dict,
		padding_idx,
		emb_size,
		hid_size,
		rnn_cell_type,
		num_layers,
		dropout
	):
		self.dict = decoder_dict
		self.vocab_size = self.dict.size()
		self.padding_idx = padding_idx
		self.emb_size = emb_size
		self.hid_size = hid_size
		self.rnn_cell_type = rnn_cell_type
		self.num_layers = num_layers
		self.dropout = dropout

		super(NaiveDecoder, self).__init__()

		self.emb = nn.Embedding(
			self.vocab_size,
			self.emb_size,
			self.padding_idx
		)

		self.stack_rnn = StackedRNNCell(
			self.num_layers,
			self.emb_size,
			self.rnn_cell_type,
			self.hid_size,
			self.dropout
		)

	def forward(self, input, dec_init):
		"""Forward compute method

		Args
		----------
		input    : N x L
		dec_init : tuple (N x H, N x H) / nL x N x H

		Return
		----------
		dec_hids : L x N x H
		dec_prev  : tuple / dec_nL x N x dec_H

		"""
		input = input[:, :-1]
		input_emb = self.emb(input) # N x (dec_L - 1) x Embz
		# if type(dec_init) == tuple:
		# 	dec_h_0, dec_c_0 = dec_init
		# else:
		# 	dec_h_0 = dec_init
		dec_prev = dec_init
		dec_hids = []
		for curr_input in input_emb.split(1, dim=1): # time step / seq_len
			curr_input = curr_input.squeeze(0) # N x Embz
			curr_output, dec_prev = self.stack_rnn(curr_input, dec_prev)
			dec_hids.append(curr_output)

		dec_hids = torch.stack(dec_hids, dim=0).transpose(0, 1) # N x L x H
		return dec_hids, dec_prev


class BahdahnauAttentiveDecoder(nn.Module):
	"""A decoder class with the Bahdahnau attention mechanism
	"""
	def __init__(
		self,
		decoder_dict,
		padding_idx,
		dec_emb_size,
		dec_hid_size,
		dec_rnn_cell_type,
		dec_num_layers,
		dropout,
		enc_hid_size,
		enc_bidirectional
	):
		self.dict = decoder_dict
		self.vocab_size = self.dict.size()
		self.padding_idx = padding_idx
		self.emb_size = dec_emb_size
		self.hid_size = dec_hid_size
		self.rnn_cell_type = dec_rnn_cell_type
		self.num_layers = dec_num_layers
		self.dropout = dropout
		self.enc_hid_size = enc_hid_size
		self.enc_bidirectional = enc_bidirectional
		if self.enc_bidirectional:
			self.enc_num_dirs = 2
		else:
			self.enc_num_dirs = 1

		if self.rnn_cell_type != 'gru':
			assert False, "In Bahdahnau attention, rnn should be gru"

		super(BahdahnauAttentiveDecoder, self).__init__()

		self.emb = nn.Embedding(
			self.vocab_size,
			self.emb_size,
			self.padding_idx
		)

		self.project_dec_input = nn.Linear(
			self.emb_size,
			self.hid_size
		)

		self.project_context = nn.Linear(
			self.enc_hid_size * self.enc_num_dirs,
			self.hid_size
		)

		self.attn = BahdahnauAttention(
			self.enc_hid_size,
			self.enc_num_dirs,
			self.hid_size
		)

		self.rnn_stack = StackedRNNCell(
			self.num_layers,
			self.emb_size,
			self.rnn_cell_type,
			self.hid_size,
			self.dropout
		)

	def forward(self, input, dec_init, enc_hids):
		"""Forward compute method

		Args
		----------
		input     : N x dec_L
		dec_init  : dec_nL x N x dec_H
		enc_hids  : N x enc_L x enc_H

		Return
		----------
		dec_hids  : N x (dec_L - 1) x dec_H
		atts      : N x enc_L x (dec_L - 1)
		dec_prev  : dec_nL x N x dec_H

		"""
		input = input[:, :-1] # N x (dec_L - 1)
		dec_prev = dec_init # dec_nL x N x dec_H
		# print(dec_prev.size())
		dec_hids_t = dec_init[-1, :, :] # N x dec_H
		dec_hids_lst = []
		att_lst = []
		for input_t in input.split(1, dim=1): # N x 1
			input_t = input_t.squeeze(1) # N
			input_t_emb = self.emb(input_t) # N x Embz
			c_curr, att_curr = self.attn(dec_hids_t, enc_hids) # N x enc_H, N x enc_L
			input_projected = self.project_dec_input(input_t_emb) # N x dec_H
			c_curr_projected = self.project_context(c_curr) # N x dec_H
			input_merged = input_projected + c_curr_projected # N x dec_H
			dec_hids_t, dec_prev = self.rnn_stack(input_merged, dec_prev)
			
			att_lst.append(att_curr)
			dec_hids_lst.append(dec_hids_t)

		# dec_hids_lst : (dec_L - 1) - N x dec_H
		# att_lst      : (dec_L - 1) - N x enc_L
		dec_hids = torch.stack(dec_hids_lst, dim=0).transpose(0, 1) # N x (dec_L - 1) x dec_H
		# print(dec_hids.size())
		atts = torch.stack(att_lst, dim=0).transpose(0, 1).transpose(1, 2) # N x enc_L x dec_L - 1

		return dec_hids, atts, dec_prev


class GlobalAttentiveDecoder(nn.Module):
	"""A decoder class with the global attention mechanism
	"""
	def __init__(
		self,
		decoder_dict,
		padding_idx,
		dec_emb_size,
		dec_hid_size,
		dec_rnn_cell_type,
		dec_num_layers,
		dropout,
		enc_hid_size,
		global_attention_type,
		enc_bidirectional
	):
		self.dict = decoder_dict
		self.vocab_size = self.dict.size()
		self.padding_idx = padding_idx
		self.emb_size = dec_emb_size
		self.hid_size = dec_hid_size
		self.rnn_cell_type = dec_rnn_cell_type
		self.num_layers = dec_num_layers
		self.dropout = dropout
		self.enc_hid_size = enc_hid_size
		self.global_attention_type = global_attention_type
		self.enc_bidirectional = enc_bidirectional
		if self.enc_bidirectional:
			self.enc_num_dirs = 2
		else:
			self.enc_num_dirs = 1
		super(GlobalAttentiveDecoder, self).__init__()

		self.emb = nn.Embedding(
			self.vocab_size,
			self.emb_size,
			self.padding_idx
		)

		self.rnn_stack = StackedRNNCell(
			self.num_layers,
			self.emb_size,
			self.rnn_cell_type,
			self.hid_size,
			self.dropout
		)

		self.attn = GlobalAttention(
			self.global_attention_type,
			self.enc_hid_size,
			self.enc_num_dirs,
			self.hid_size
		)

	def forward(self, input, dec_init, enc_hids):
		"""Forward compute method of global attentive decoder

		Args
		----------
		input     : N x dec_L
		dec_init  : tuple / dec_nL x N x dec_H
		enc_hids  : N x enc_L x enc_H

		Reture
		----------
		dec_hids  : N x (dec_L - 1) x dec_H
		atts      : N x enc_L x (dec_L - 1)
		dec_prev  : tuple / dec_nL x N x dec_H
		"""
		dec_prev = dec_init
		# # FOLLOWING might be a BUG
		# if self.rnn_cell_type == 'lstm':
		# 	dec_h_prev, dec_c_prev = dec_prev
		# else:
		# 	dec_h_prev = dec_prev

		input = input[:, :-1] # N x (dec_L - 1)
		dec_hids_lst = []
		att_lst = []
		for input_t in input.split(1, dim=1): # (decL - 1) - N x 1
			input_t = input_t.squeeze(1) # N
			input_t_emb = self.emb(input_t) # N x Embz
			# print(input_t_emb.size())
			# print(dec_prev.size())
			dec_h_curr, dec_prev = self.rnn_stack(input_t_emb, dec_prev)
			h_att_curr, att_curr = self.attn(dec_h_curr, enc_hids)

			dec_hids_lst.append(h_att_curr)
			att_lst.append(att_curr)

		dec_hids = torch.stack(dec_hids_lst).transpose(0, 1) # N x (dec_L - 1) x dec_H
		atts = torch.stack(att_lst).transpose(0, 1).transpose(1, 2)

		return dec_hids, atts, dec_prev