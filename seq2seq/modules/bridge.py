import torch
import torch.nn as nn
import torch.nn.functional as F

class Bridge(nn.Module):
	"""A bridge module which connects encoder's output/last states
	to decoder's input/initial states.

	Currently, a multi-layer perceptron is implemented to map between
	the encoder and the decoder

	"""
	def __init__(
		self,
		encoder_bidirectioinal,
		encoder_num_layers,
		encoder_hid_size,
		encoder_rnn_cell_type,
		decoder_num_layers,
		decoder_hid_size,
		decoder_rnn_cell_type
	):
		self.encoder_bidirectioinal = encoder_bidirectioinal
		if self.encoder_bidirectioinal:
			self.encoder_num_dirs = 2
		else:
			self.encoder_num_dirs = 1
		self.encoder_num_layers = encoder_num_layers
		self.encoder_hid_size = encoder_hid_size
		self.encoder_rnn_cell_type = encoder_rnn_cell_type

		self.decoder_num_layers = decoder_num_layers
		self.decoder_hid_size = decoder_hid_size
		self.decoder_rnn_cell_type = decoder_rnn_cell_type

		if self.encoder_rnn_cell_type is not self.decoder_rnn_cell_type:
			assert(False, "encoder, decoder rnn cell type must match")

		self.bridge = nn.Linear(
			self.encoder_num_dirs * self.encoder_hid_size,
			self.decoder_num_layers * self.decoder_hid_size
		)

		if self.encoder_rnn_cell_type == 'lstm':
			self.bridge_c = nn.Linear(
				self.encoder_num_dirs * self.encoder_hid_size,
				self.decoder_num_layers * self.decoder_hid_size
			)

	def forward(self, enc_last):
		"""Forward compute method

		Args
		----------
		enc_last : tuple / (enc_L x enc_D) x N x enc_H

		Return
		----------
		dec_init : tuple / dec_L x N x dec_H

		"""
		if type(enc_last) == tuple: # 'lstm encoder'
			enc_h_n, enc_c_n = enc_last
			enc_c_n = enc_c_n.contiguous().view(
				self.encoder_num_layers,
				self.encoder_num_dirs,
				-1,
				self.encoder_hid_size
			).transpose(1, 2)[-1]
			enc_c_n = enc_c_n.contiguous().view(
				-1,
				self.encoder_num_dirs * self.encoder_hid_size
			)
			dec_c_0 = F.tanh(self.bridge_c(enc_c_n))
			dec_c_0 = dec_c_0.contiguous().view(
				-1,
				self.decoder_num_layers,
				self.decoder_hid_size
			).transpose(0, 1)
			dec_c_0 = dec_c_0.contiguous().view(
				self.decoder_num_layers,
				-1,
				self.decoder_hid_size
			)
		else:
			enc_h_n = enc_last

		enc_h_n = enc_h_n.contiguous().view(
			self.encoder_num_layers,
			self.encoder_num_dirs,
			-1,
			self.encoder_hid_size
		).transpose(1, 2)[-1] # N x enc_D x enc_H
		enc_h_n = enc_h_n.contiguous().view(
			-1,
			self.encoder_num_dirs * self.encoder_hid_size
		) # N x (enc_D x enc_H)
		dec_h_0 = F.tanh(self.bridge(enc_h_n)) # N x (dec_L x dec_H)
		dec_h_0 = dec_h_0.contiguous().view(
			-1,
			self.decoder_num_layers,
			self.decoder_hid_size
		).transpose(0, 1)
		dec_h_0 = dec_h_0.contiguous().view(
			self.decoder_num_layers,
			-1,
			self.decoder_hid_size
		) # dec_L x N x dec_H

		if type(enc_last) == tuple:
			return (dec_h_0, dec_c_0)
		else:
			return dec_h_0

