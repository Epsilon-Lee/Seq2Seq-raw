import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedRNNCell(nn.Module):
	"""Stacked RNN Cell, which is used for
	one-step computation forward.
	"""
	def __init__(
		self,
		num_layers,
		emb_size,
		rnn_cell_type,
		hid_size
	):
		self.num_layers = num_layers
		self.emb_size = emb_size
		self.rnn_cell_type = rnn_cell_type
		self.hid_size = hid_size
		
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
		elif: self.rnn_cell_type == 'lstm':
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
					self.GRUCell(
						self.emb_size,
						self.hid_size
					)
				)
	
	def forward(self, input, prev):
		"""Forward compute for each time-step

		Args
		----------
		input  : N x Embz torch.FloatTensor
		prev   : tuple (prev_h, prev_c) / L x N x H

		Return
		----------
		output : N x H
		curr   : tuple (curr_h, curr_c)/ L x N x H

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
				curr_h.append(layer_input)

		if type(prev) == tuple:
			return layer_input, (curr_h, curr_c)
		else:
			return layer_input, curr_h

class Decoder(nn.Module):
	