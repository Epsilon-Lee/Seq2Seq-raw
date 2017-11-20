import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.encoder import Encoder
from modules.decoder import NaiveDecoder, BahdahnauAttentiveDecoder, GlobalAttentiveDecoder
from modules.bridge import Bridge
from modules.generator import Generator


class Seq2Seq(nn.Module):
	"""A naive sequence-to-sequence model
	"""
	def __init__(
		self,
		encoder_dict,
		encoder_padding_idx,
		encoder_emb_size,
		encoder_hid_size,
		encoder_bidirectional,
		encoder_rnn_cell_type,
		encoder_is_packed,
		encoder_batch_first,
		encoder_num_layers,
		encoder_dropout,
		decoder_dict,
		decoder_padding_idx,
		decoder_emb_size,
		decoder_hid_size,
		decoder_rnn_cell_type,
		decoder_num_layers,
		decoder_dropout,
		generator_dim_lst,
		generator_num_layers
	):
		super(Seq2Seq, self).__init__()

		self.encoder = Encoder(
			encoder_dict,
			encoder_padding_idx,
			encoder_emb_size,
			encoder_hid_size,
			encoder_bidirectional,
			encoder_rnn_cell_type,
			encoder_is_packed,
			encoder_batch_first,
			encoder_num_layers,
			encoder_dropout
		)

		self.bridge = Bridge(
			encoder_bidirectional,
			encoder_num_layers,
			encoder_hid_size,
			encoder_rnn_cell_type,
			decoder_num_layers,
			decoder_hid_size,
			decoder_rnn_cell_type
		)

		self.decoder = NaiveDecoder(
			decoder_dict,
			decoder_padding_idx,
			decoder_emb_size,
			decoder_hid_size,
			decoder_rnn_cell_type,
			decoder_num_layers,
			decoder_dropout
		)

		self.generator = Generator(
			decoder_dict.size(),
			decoder_hid_size,
			generator_dim_lst,
			generator_num_layers
		)

	def forward(
		self,
		input,
		input_mask,
		input_lengths,
		output,
		output_mask,
		output_lengths
	):
		"""Forward compute method for the naive seq2seq model

		Args
		----------

		Return
		----------

		"""
		enc_hids, enc_last = self.encoder(
			input,
			input_mask,
			input_lengths
		)

		dec_init = self.bridge(enc_last)

		dec_hids = self.decoder(output, dec_init) # N x (dec_L - 1) x H

		preds = self.generator(dec_hids) # N x (dec_L - 1) x vocab_size

		return preds


class BahdahnauAttentionSeq2Seq(nn.Module):
	"""A sequence-to-sequence model using Bahdahnau attention
	"""
	def __init__(
		self,
		encoder_dict,
		encoder_padding_idx,
		encoder_emb_size,
		encoder_hid_size,
		encoder_bidirectional,
		encoder_rnn_cell_type,
		encoder_is_packed,
		encoder_batch_first,
		encoder_num_layers,
		encoder_dropout,
		decoder_dict,
		decoder_padding_idx,
		decoder_emb_size,
		decoder_hid_size,
		decoder_rnn_cell_type,
		decoder_num_layers,
		decoder_dropout,
		generator_dim_lst,
		generator_num_layers
	):
		super(BahdahnauAttentionSeq2Seq, self).__init__()

		self.encoder = Encoder(
			encoder_dict,
			encoder_padding_idx,
			encoder_emb_size,
			encoder_hid_size,
			encoder_bidirectional,
			encoder_rnn_cell_type,
			encoder_is_packed,
			encoder_batch_first,
			encoder_num_layers,
			encoder_dropout
		)

		self.bridge = Bridge(
			encoder_bidirectional,
			encoder_num_layers,
			encoder_hid_size,
			encoder_rnn_cell_type,
			decoder_num_layers,
			decoder_hid_size,
			decoder_rnn_cell_type
		)

		self.decoder = BahdahnauAttentiveDecoder(
			decoder_dict,
			decoder_padding_idx,
			decoder_emb_size,
			decoder_hid_size,
			decoder_rnn_cell_type,
			decoder_num_layers,
			decoder_dropout,
			encoder_hid_size,
			encoder_bidirectional
		)

		self.generator = Generator(
			decoder_dict.size(),
			decoder_hid_size,
			generator_dim_lst,
			generator_num_layers
		)

	def forward(
		self,
		input,
		input_mask,
		input_lengths,
		output,
		output_mask,
		output_lengths
	):
		"""Forward compute method

		Args
		----------
		input          : N x enc_L
		input_mask     : N x enc_L
		input_lengths  : list len() => N
		output         : N x dec_L
		output_mask    : N x dec_L
		output_lengths : list len() => N

		Return
		----------
		preds          : N x (dec_L - 1) x vocab_size

		atts           : N x enc_L x (dec_L - 1)

		"""
		enc_hids, enc_last = self.encoder(
			input,
			input_mask,
			input_lengths
		)

		# print('type enc_hids: %s' % str(type(enc_hids)))

		dec_init = self.bridge(enc_last)

		dec_hids, atts = self.decoder(output, dec_init, enc_hids)

		preds = self.generator(dec_hids)

		return preds, atts


class GlobalAttentionSeq2Seq(nn.Module):
	"""A sequence-to-sequence model using global attention
	"""
	def __init__(
		self,
		encoder_dict,
		encoder_padding_idx,
		encoder_emb_size,
		encoder_hid_size,
		encoder_bidirectional,
		encoder_rnn_cell_type,
		encoder_is_packed,
		encoder_batch_first,
		encoder_num_layers,
		encoder_dropout,
		decoder_dict,
		decoder_padding_idx,
		decoder_emb_size,
		decoder_hid_size,
		decoder_rnn_cell_type,
		decoder_num_layers,
		decoder_dropout,
		global_attention_type,
		generator_dim_lst,
		generator_num_layers
	):
		super(GlobalAttentionSeq2Seq, self).__init__()

		self.encoder = Encoder(
			encoder_dict,
			encoder_padding_idx,
			encoder_emb_size,
			encoder_hid_size,
			encoder_bidirectional,
			encoder_rnn_cell_type,
			encoder_is_packed,
			encoder_batch_first,
			encoder_num_layers,
			encoder_dropout
		)

		self.bridge = Bridge(
			encoder_bidirectional,
			encoder_num_layers,
			encoder_hid_size,
			encoder_rnn_cell_type,
			decoder_num_layers,
			decoder_hid_size,
			decoder_rnn_cell_type
		)

		self.decoder = GlobalAttentiveDecoder(
			decoder_dict,
			decoder_padding_idx,
			decoder_emb_size,
			decoder_hid_size,
			decoder_rnn_cell_type,
			decoder_num_layers,
			decoder_dropout,
			encoder_hid_size,
			global_attention_type
		)

		self.generator = Generator(
			decoder_dict.size(),
			decoder_hid_size,
			generator_dim_lst,
			generator_num_layers
		)

	def forward(
		self,
		input,
		input_mask,
		input_lengths,
		output,
		output_mask,
		output_lengths
	):
		"""Forward compute method

		Args
		----------
		input          : N x enc_L
		input_mask     : N x enc_L
		input_lengths  : list len() => N
		output         : N x dec_L
		output_mask    : N x dec_L
		output_lengths : list len() => N

		Return
		----------
		preds          : N x (dec_L - 1) x vocab_sizes

		atts           : N x enc_L x (dec_L - 1)

		"""
		enc_hids, enc_last = self.encoder(
			input,
			input_mask,
			input_lengths
		)

		dec_init = self.bridge(enc_last)

		dec_hids, atts = self.decoder(output, dec_init, enc_hids)

		voc_dstrs = self.generator(dec_hids) # N x (dec_L - 1) x vocab_size

		return preds, atts

