import torch
import torch.nn as nn
from nn.utils.rnn import pack_padded_sequence as pack
from nn.utils.rnn import pad_packed_sequence as unpack

class Encoder(nn.Module):
	"""Encoder class which processes sequences of input
	text and produces:
		1). enc_hids
		2). enc_h_n, enc_c_n (if use LSTM)
	"""

	def __init__(
		self,
		encoder_dict,
		emb_size,
		hid_size,
		bidirectional,
		rnn_cell_type,
		is_packed,
		batch_first,
		num_layers,
		dropout
	):
		self.dict = encoder_dict
		self.vocab_size = src_dict.size()
		self.emb_size = emb_size
		self.hid_size = hid_size
		self.bidirectional = bidirectional
		if self.bidirectional:
			self.num_dirs = 2
		else:
			self.num_dirs = 1
		# 'rnn', 'gru', 'lstm'
		self.rnn_cell_type = rnn_cell_type
		self.is_packed = is_packed
		self.batch_first = batch_first
		self.num_layers = num_layers
		self.dropout = dropout
		
		self.emb = nn.Embedding(
			self.vocab_size,
			self.emb_size,
			self.dict.src_specials['<pad>']
		)

		if self.rnn_cell_type == 'rnn':
			self.rnn = nn.RNN(
				self.emb_size,
				self.hid_size,
				self.num_layers,
				batch_first=self.batch_first,
				dropout=self.dropout,
				bidirectional=self.bidirectional
			)
		elif: self.rnn_cell_type == 'lstm':
			self.rnn = nn.LSTM(
				self.emb_size,
				self.hid_size,
				self.num_layers,
				batch_first=self.batch_first,
				dropout=self.dropout,
				bidirectional=self.bidirectional
			)
		else:
			self.rnn = nn.GRU(
				self.emb_size,
				self.hid_size,
				self.num_layers,
				batch_first=self.batch_first,
				dropout=self.dropout,
				bidirectional=self.bidirectional
			)

	def forward(self, input, input_mask, input_lengths):
		"""Forward compute method

		Args
		----------
		input         : N x L torch.LongTensor
		input_mask    : N x L torch.FloatTensor
		input_lengths : N python list [int]

		Return
		----------
		enc_hids   : N x L x H or L x N x H torch.FloatTensor
		enc_last   : tuple / (n_layer x n_dir) x N x H

		"""
		if not batch_first:
			input = input.t() # L x N
		input_emb = self.emb(input) # N x L x H / L x N x H
		if self.is_packed:
			input_emb = pack(
				input_emb,
				input_lengths,
				self.batch_first
			)
		
		enc_hids, enc_last = self.rnn(input_emb)
		
		if self.is_packed:
			enc_hids = unpack(
				enc_hids,
				self.batch_first
			)

		return enc_hids, enc_last