import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


# multi-layer bi-directional lstm encoder
class Encoder(nn.Module):
	def __init__(self, opts, srcDictionary):
		super(Encoder, self).__init__()
		self.opts = opts
		self.emb = nn.Embedding(srcDictionary.size(), opts.embSize, padding_idx=srcDictionary.src_specials['<pad>'])
		self.rnn = nn.LSTM(opts.embSize, opts.hidSize, opts.layerNum, bidirectional=opts.bidirectional)
 
	def selectLastStates(self, output, batchLengths):
		# output: seq_len x bz x (hz x num_directions)
		# batchLengths: a list of length for each example
		seq_len = output.size(0)
		selectIdxs = [length - 1 + i * seq_len for i, length in enumerate(batchLengths)]
		if self.opts.cuda:
			selectIdxs = torch.LongTensor(selectIdxs).cuda()
		output = output.transpose(0, 1).contiguous().view(-1, output.size(2)) # (bz x seq_len) x (hz x num_directions)
		return torch.index_select(output, 0, Variable(selectIdxs)) # bz x (hz x num_directions)

	def forward(self, inputs, packed=True):
		batchIdxPt, batchMaskPt, batchLengths = inputs
		batchIdxPt = batchIdxPt.t().contiguous()
		batchLengths = list(batchLengths)
		# print batchIdxPt
		# print batchLengths
		# print len(batchLengths)
		batchEmb = self.emb(batchIdxPt)
		if packed:
			batchEmb_packed = pack_padded_sequence(batchEmb, batchLengths)
			output_packed, (h_n, c_n )= self.rnn(batchEmb_packed)
			output, _ = pad_packed_sequence(output_packed)
		else:
			output, (h_n, c_n) = self.rnn(batchEmb)
		# return output, h_n, c_n # seq_len x bz x (hz x num_directions), (num_layers x num_directions) x bz x hz
		lastStates = self.selectLastStates(output, batchLengths)
		return lastStates # bz x (hz x num_directions)


# multi-layer lstm stack: motivated by OpenNMT-py
class StackedLSTM(nn.Module):
	def __init__(self, opts):
		super(StackedLSTM, self).__init__()
		if opts.bidirectional == True:
			self.dirNum = 2
		else:
			self.dirNum = 1
		# self.rnnStack = nn.ModuleList([nn.LSTMCell(opts.embSize, opts.hidSize * dirNum)] \
		# 				+ [nn.LSTMCell(opts.hidSize, opts.hidSize * dirNum) for layer in range(opts.layerNum - 1)])
		self.rnnStack = nn.ModuleList([nn.LSTMCell(opts.embSize, opts.hidSize * self.dirNum)] \
						+ [nn.LSTMCell(opts.hidSize * self.dirNum, opts.hidSize * self.dirNum) \
						for layer in range(opts.layerNum - 1)])
	
	def forward(self, emb_t, hids):
		# emb_t: bz x embz
		# hids: a list of tuple (h_0, c_0)
		output_l = emb_t
		new_hids = []
		for lstmCell, hid in zip(self.rnnStack, hids):
			# print output_l.size()
			# print hid[0].size()
			# print lstmCell
			h_1, c_1 = lstmCell(output_l, hid)
			output_l = h_1
			new_hids += [(h_1, c_1)]
		return output_l, new_hids


# multi-layer lstm decoder
class Decoder(nn.Module):
	def __init__(self, opts, tgtDictionary):
		super(Decoder, self).__init__()
		self.emb = nn.Embedding(tgtDictionary.size(), opts.embSize, \
								padding_idx=tgtDictionary.tgt_specials['<pad>'])
		self.rnnStack = StackedLSTM(opts)

	def forward(self, inputs, encoderHid):
		# inputs: batchIdxPt, batchMask, batchLengths
		# encoderHid: list of tuple with length opts.layerNum
		batchIdxPt, _, _ = inputs
		batchIdxPt = batchIdxPt.t() # seq_len x bz
		batchEmb = self.emb(batchIdxPt[:-1]) # exclude '</s>' idx
		outputs = []
		hids = encoderHid
		for batchEmb_t in batchEmb.split(1): # split size: 1
			batchEmb_t = batchEmb_t.squeeze(0) # bz x embz
			# print batchEmb_t.size()
			output_t, hids = self.rnnStack(batchEmb_t, hids)
			# print 'output_t', output_t.size()
			outputs += [output_t.unsqueeze(0)]
		outputs = torch.cat(outputs) # (seq_len - 1) x bz x hz

		return outputs


class AttentiveDecoder(nn.Module):
	def __init__(self, opts):
		pass

# Composite models
# 1. seq2seq model
# 2. seq2seq+att model

class Seq2Seq(nn.Module):
	def __init__(self, opts, srcDictionary, tgtDictionary):
		super(Seq2Seq, self).__init__()
		self.encoder = Encoder(opts, srcDictionary)
		self.decoder = Decoder(opts, tgtDictionary)
		dirNum = 1
		if opts.bidirectional:
			dirNum = 2
		self.generator = nn.Linear(opts.hidSize * dirNum, tgtDictionary.size())
		self.logSoftmax = nn.LogSoftmax()
		weight = torch.ones(tgtDictionary.size())
		weight[tgtDictionary.tgt_specials['<pad>']] = 0
		self.loss = nn.NLLLoss(weight, size_average=False)
		if opts.cuda:
			self.loss.cuda()

	# Initialize weights
	def init_weight(self):
		initrange = 0.1
		# self.encoder.emb.weight.data.uniform_(-initrange, initrange)
		# self.decoder.emb.weight.data.uniform_(-initrange, initrange)
		for p in self.parameters():
			p.data.uniform_(-initrange, initrange)

	def forward(self, srcInputs, tgtInputs, opts):
		encoderHid = self.encoder(srcInputs, opts.packed) # bz x (hz x num_directions)
		encoderHidLst = [(encoderHid, encoderHid) for i in range(opts.layerNum)]
		outputs = self.decoder(tgtInputs, encoderHidLst) # (seq_len - 1) x bz x hz
		# print outputs.size()
		outputs = outputs.view(-1, outputs.size(2))
		unnormalized_probs = self.generator(outputs)#.view(-1, outputs.size(2))
		# probs = F.softmax(unnormalized_probs) # ((seq_len - 1) x bz) x hz
		logProbs = self.logSoftmax(unnormalized_probs)
		return logProbs

	def MLELoss(self, logProbs, tgtInputs):
		tgts = tgtInputs[0].t()[1:].contiguous().view(-1) # ((seq_len - 1) x bz)
		loss_batch = self.loss(logProbs, tgts)
		return loss_batch

	def MemoryEfficientMLELoss(self, probs, tgtInputs):
		pass


class Seq2Seq_MaximumEntropy(nn.Module):
	def __init__(self, opts, src_dict, tgt_dict):
		"""Initialize Model"""
		super(Seq2Seq_MaximumEntropy, self).__init__()

		self.use_gpu = opts.use_gpu

		self.bidirectional = opts.bidirectional
		self.src_num_directions = 2 if self.bidirectional else 1
		self.src_vocab_size = src_dict.size()
		self.tgt_vocab_size = tgt_dict.size()
		self.src_emb_size = opts.src_emb_size
		self.tgt_emb_size = opts.tgt_emb_size
		self.src_hid_size = opts.src_hid_dim // self.src_num_directions
		self.tgt_hid_size = opts.tgt_hid_dim
		self.src_num_layers = opts.src_num_layers
		self.tgt_num_layers = opts.tgt_num_layers

		self.batch_first = opts.batch_first
		self.dropout = opts.dropout

		self.src_dict = src_dict
		self.tgt_dict = tgt_dict

		self.src_emb = nn.Embedding(
			self.src_vocab_size,
			self.src_emb_size,
			self.src_dict.src_specials['<pad>']
		)

		self.tgt_emb = nn.Embedding(
			self.tgt_vocab_size,
			self.tgt_emb_size,
			self.tgt_dict.tgt_specials['<pad>']
		)

		self.encoder = nn.LSTM(
			self.src_emb_size,
			self.src_hid_size,
			self.src_num_layers,
			batch_first=self.batch_first,
			dropout=self.dropout,
			bidirectional=self.bidirectional
		)

		self.enchid_to_dechid = nn.Linear(
			self.src_hid_size * self.src_num_directions,
			self.tgt_hid_size * self.tgt_num_layers * 2
		)

		self.decoder = nn.LSTM(
			self.tgt_emb_size,
			self.tgt_hid_size,
			self.tgt_num_layers,
			batch_first=self.batch_first,
			dropout=self.dropout,
		)

		self.decoder_transform = nn.Linear(
			self.tgt_hid_size,
			self.tgt_hid_size
		)

		self.readout = nn.Linear(
			self.tgt_hid_size,
			self.tgt_vocab_size
		)

		weight_mask = torch.ones(tgt_dict.size())
		weight_mask[tgt_dict.tgt_specials['<pad>']] = 0
		self.criterion = nn.CrossEntropyLoss(weight=weight_mask, size_average=False) # deal with unnormalized prob.

	def init_weight(self):
		initrange = 0.1
		self.src_emb.weight.data.uniform_(-initrange, initrange)
		self.tgt_emb.weight.data.uniform_(-initrange, initrange)
		self.enchid_to_dechid.bias.data.fill_(0)
		self.decoder_transform.bias.data.fill_(0)
		self.readout.bias.data.fill_(0)

	def forward(self, src_input, tgt_input):
		src_idxs, _ , _ = src_input
		tgt_idxs, _ , _ = tgt_input

		src_embs = self.src_emb(src_idxs)
		_ , (enc_h_n, enc_c_n) = self.encoder(src_embs) # enc_h_n: (nLayer x nDir) x bz x hz
		enc_h_n = enc_h_n.view(-1, self.src_num_directions, enc_h_n.size(1), enc_h_n.size(2)).transpose(1, 2)
		enc_h_n = enc_h_n.contiguous().view(enc_h_n.size(0), enc_h_n.size(1), -1)
		enc_h_n_last_layer = enc_h_n[self.src_num_directions - 1].contiguous()
		dec_hc_0 = self.enchid_to_dechid(enc_h_n_last_layer) # bz x (nL x hz x 2)
		dec_hc_0 = F.tanh(dec_hc_0)
		dec_hc_0 = dec_hc_0.view(dec_hc_0.size(0), -1, self.tgt_hid_size)
		dec_hc_0 = dec_hc_0.transpose(0, 1)
		dec_hc_0 = dec_hc_0.contiguous().view(2, self.tgt_num_layers, dec_hc_0.size(1), self.tgt_hid_size)
		dec_h_0, dec_c_0 = dec_hc_0.split(1) # dec_h_0: 1 x nL x bz x hz
		# print(dec_hc_0.size(), dec_h_0.size(), dec_c_0.size())
		dec_h_0 = dec_h_0.squeeze(0) # nL x bz x hz
		dec_c_0 = dec_c_0.squeeze(0)

		tgt_idxs = tgt_idxs[:, :-1].contiguous()
		# print(tgt_idxs.size())
		tgt_embs = self.tgt_emb(tgt_idxs)
		dec_outputs, _ = self.decoder(tgt_embs, (dec_h_0, dec_c_0)) # bz x seq_len-1 x hz
		dec_out_transformed = self.decoder_transform(dec_outputs)
		dec_logit = self.readout(dec_out_transformed) # bz x seq_len-1 x hz

		return dec_logit # unnormalized prob.

	def MLELoss(self, dec_logit, tgt_input):
		tgt = tgt_input[0][:, 1:].contiguous()
		return self.criterion(
			dec_logit.view(
				dec_logit.size(0) * dec_logit.size(1),
				-1
			),
			tgt.view(-1)
		)

	def greedy_search(self, src_id, max_decode_length):
		"""Batch mode greedy search to get the prediction

		Batch mode greedy search try to predict the current word with the maximum probability
		at the current predicted distribution over vocab. And then feed to the input at next
		timestep to predict next word. 

		Argument
		----------
		src_id : Variable(torch.LongTensor)
			source side batch with the shape: N x Len; this is padded with id src_dict.src_specials['<pad>']. 

		Return
		----------
		score   : torch.FloatTensor


		pred_id : torch.LongTensor
			predicted target side word id seq list with the shape: N x MaxLen; this is not padded but
			the EOS can be find by searching for the id tgt_dict.tgt_specials['</s>'] in each row. 
		"""
		self.eval()
		# get encoder final state
		if self.use_gpu:
			src_id = src_id.cuda()
		src_id_emb = self.src_emb(src_id) # N x seq_len x embz
		src_context, (src_h_n, src_c_0) = self.encoder(src_id_emb) # src_h_n: (n_layers x n_dir) x N x hz_src
		src_h_n = src_h_n.view(
			self.src_num_layers,
			self.src_num_directions,
			-1,
			self.src_hid_size
		)[self.src_num_layers - 1] # n_dir x N x hz_src
		src_h_n = src_h_n.transpose(0, 1).contiguous().view(
			-1,
			self.src_num_directions * self.src_hid_size
		) # N x (n_dir x hz_src) 
		tgt_hc_0 = self.enchid_to_dechid(src_h_n) # N x (2 x hz_tgt)
		tgt_h_0, tgt_c_0 = tgt_hc_0.contiguous().view(
			-1,
			2,
			self.tgt_hid_size
		).transpose(0, 1).split(1) # tgt_h_0: 1 x N x hz_tgt

		# decode until max_decode_length arrived
		batch_size = src_id.size(0) # int
		# `cur` means current
		cur_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.tgt_dict.tgt_specials['<s>'])) # N x 1
		if self.use_gpu:
			cur_input = cur_input.cuda()
		h_n, c_n = tgt_h_0.contiguous(), tgt_c_0.contiguous()
		score = [] # to store negative log likelihood for every predicted id
		pred = [] # to store decoded id
		for time_step in xrange(max_decode_length):
			cur_input_emb = self.tgt_emb(cur_input) # N x 1 x embz_tgt
			cur_output, (h_n, c_n) = self.decoder(cur_input_emb, (h_n, c_n)) # cur_output: N x 1 x hz_tgt
			cur_output = cur_output.squeeze(1) # N x hz_tgt
			cur_output_transformed = self.decoder_transform(cur_output) # N x hz_tgt
			cur_readout = self.readout(cur_output_transformed) # N x vocab_size_tgt
			cur_score, cur_pred = torch.max(cur_readout.data, 1) # cur_score: N, torch.FloatTensor
			score += [cur_score]
			pred += [cur_pred]
			# make cur_input from cur_pred
			cur_input = Variable(cur_pred.unsqueeze(1)) # N x 1
			if self.use_gpu:
				cur_input = cur_input.cuda()
		score = torch.stack(score, 0).t() # N x max_decode_length, torch.FloatTensor
		pred = torch.stack(pred, 0).t() # N x max_decode_length, torch.LongTensor
		self.train()
		return score, pred

	def beam_search_per_example(self, src_id, beam_size, max_decode_length):
		"""Beam search decoding in per-example mode

		Argument
		----------
		src_id : Variable(torch.LongTensor)
			src_id has shape: N x seq_len

		beam_size : int
			beam size

		max_decode_length : int
			the maximum number of steps to decode

		Return
		----------
		score : list
			a list of float numbers which relects the total log-likelihood score
			for each example, list has the length of N x beam_size

		pred  : torch.LongTensor
			the size of the tensor is N x beam_size x max_decode_length
		"""
		self.eval()
		if self.use_gpu:
			src_id = src_id.cuda()
		# encoder computation
		src_emb = self.src_emb(src_id) # N x seq_len x embz
		src_context, (src_h_n, src_c_0) = self.encoder(src_emb) # src_h_n: (nL x nDir) x N x hz
		src_h_n = src_h_n.view(
			self.src_num_layers,
			self.src_num_directions,
			-1,
			self.src_hid_size
		).transpose(1, 2).contiguous().view(
			self.src_num_layers,
			-1,
			self.src_num_directions * self.src_hid_size
		)[self.src_num_layers - 1] # N x (nDir x hz)
		tgt_hc_0 = self.enchid_to_dechid(src_h_n) # N x (2 x nL x hz)
		tgt_hc_0 = F.tanh(tgt_hc_0)
		tgt_hc_0 = tgt_hc_0.view(
			-1,
			2 * self.tgt_num_layers,
			self.tgt_hid_size
		).transpose() # (2 * nL) x N x hz
		tgt_h_0, tgt_c_0 = tgt_hc_0.contiguous().view(
			2,
			self.tgt_num_layers,
			-1,
			self.tgt_hid_size
		).split(1)
		tgt_h_0 = tgt_h_0.squeeze(0) # nL x N x hz
		tgt_c_0 = tgt_c_0.squeeze(0)
		scores = []
		preds = []
		for idx in range(tgt_h_0.size(1)):
			h_0 = tgt_h_0[:, idx, :].unsqueeze(1) # nL x 1 x hz
			c_0 = tgt_h_0[:, idx, :].unsqueeze(1) # nL x 1 x hz
			h_0_lst = [h_0 for i in range(beam_size)]
			c_0_lst = [c_0 for i in range(beam_size)]
			h_0 = torch.stack(h_0_lst, 1) # nL x beam_size x hz
			c_0 = torch.stack(c_0_lst, 1)

			# prepare first step forward
			cur_input = Variable(
				torch.LongTensor(
					beam_size,
					1
				).fill_(self.tgt_dict.tgt_specials['<s>'])
			) # beam_size x 1
			h_t, c_t = h_0, c_0

			# variable used to store total score and predicted sequence of each beam
			beam_score = torch.FloatTensor(beam_size).fill_(0) # beam_size
			pred_total = torch.LongTensor() # torch.LongTensor with no dimension
			for time_step in range(max_decode_length):
				# forward one-step
				cur_input_emb = self.tgt_emb(cur_input) # beam_size x 1 x embz
				cur_output, (h_tp1, c_tp1) = self.decoder(
					cur_input_emb,
					(h_t, c_t)
				) # beam_size x 1 x hz, nL x beam_size x hz, _ 
				cur_output = self.decoder_transform(cur_output)
				cur_readout = self.readout(cur_output) # beam_size x 1 x vocab_size
				cur_log_likelihood = F.log_softmax(cur_readout).data.squeeze(1) # beam_size x vocab_size
				score, pred = torch.topk(cur_log_likelihood, beam_size, dim=1) # beam_size x beam_size
				score_total = score + beam_score.unsqueeze(1) # beam_size x beam_size
				score_flatten = score.view(-1) # beam_size^2
				pred_flatten = pred.view(-1) # beam_size^2
				score_cur_beam, pred_step_idx = torch.topk(score_flatten, beam_size, dim=0) # beam_size
				pred_step = pred.index_select(pred_step_idx)
				beam_score = beam_score[pred_step_idx / beam_size]
				beam_score += score_cur_beam
				if len(pred_total.size()) is 0:
					pred_total = torch.cat([pred_total, pred_step.unsqueeze(1)], dim=1) # N x timestep --> N x (timestep + 1)
				else:
					pred_total = pred_total[pred_step_idx / beam_size, :]
					pred_total = torch.cat([pred_total, pred_step.unsqueeze(1)], dim=1)
				cur_input = pred_step
				h_t = h_t[:, pred_step_idx, :]
				c_t = c_t[:, pred_step_idx, :]
			scores += [beam_score]
			preds += [pred_total] 


	def beam_search_batch(self, src_id, max_decode_length):
		"""Batch-mode beam search decoding"""
		pass
		

class Seq2SeqPyTorch(nn.Module):
	"""Container module with an encoder, deocder, embeddings."""
	def __init__(
		self,
		src_emb_dim,
		trg_emb_dim,
		src_vocab_size,
		trg_vocab_size,
		src_hidden_dim,
		trg_hidden_dim,
		batch_size,
		pad_token_src,
		pad_token_trg,
		bidirectional=True,
		nlayers=2,
		nlayers_trg=1,
		dropout=0.
	):
		"""Initialize model."""
		super(Seq2SeqPyTorch, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.trg_vocab_size = trg_vocab_size
		self.src_emb_dim = src_emb_dim
		self.trg_emb_dim = trg_emb_dim
		self.src_hidden_dim = src_hidden_dim
		self.trg_hidden_dim = trg_hidden_dim
		# self.batch_size = batch_size
		self.bidirectional = bidirectional
		self.nlayers = nlayers
		self.dropout = dropout
		self.num_directions = 2 if bidirectional else 1
		self.pad_token_src = pad_token_src
		self.pad_token_trg = pad_token_trg
		self.src_hidden_dim = src_hidden_dim // 2 \
			if self.bidirectional else src_hidden_dim

		self.src_embedding = nn.Embedding(
			src_vocab_size,
			src_emb_dim,
			self.pad_token_src
		)
		self.trg_embedding = nn.Embedding(
			trg_vocab_size,
			trg_emb_dim,
			self.pad_token_trg
		)

		self.encoder = nn.LSTM(
			src_emb_dim,
			self.src_hidden_dim,
			nlayers,
			bidirectional=bidirectional,
			batch_first=True,
			dropout=self.dropout
		)

		self.decoder = nn.LSTM(
			trg_emb_dim,
			trg_hidden_dim,
			nlayers_trg,
			dropout=self.dropout,
			batch_first=True
		)

		self.encoder2decoder = nn.Linear(
			self.src_hidden_dim * self.num_directions,
			trg_hidden_dim
		)
		self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

		weight_mask = torch.ones(self.trg_vocab_size)
		weight_mask[self.pad_token_trg] = 0
		self.criterion = nn.CrossEntropyLoss(weight=weight_mask, size_average=False)
		
		self.init_weights()

	def init_weights(self):
		"""Initialize weights."""
		initrange = 0.1
		self.src_embedding.weight.data.uniform_(-initrange, initrange)
		self.trg_embedding.weight.data.uniform_(-initrange, initrange)
		self.encoder2decoder.bias.data.fill_(0)
		self.decoder2vocab.bias.data.fill_(0)

	def get_state(self, input):
		"""Get cell states and hidden states."""
		batch_size = input.size(0) \
			if self.encoder.batch_first else input.size(1)
		h0_encoder = Variable(torch.zeros(
			self.encoder.num_layers * self.num_directions,
			batch_size,
			self.src_hidden_dim
		))
		c0_encoder = Variable(torch.zeros(
			self.encoder.num_layers * self.num_directions,
			batch_size,
			self.src_hidden_dim
		))

		return h0_encoder.cuda(), c0_encoder.cuda()

	def forward(self, input_src, input_trg, ctx_mask=None, trg_mask=None):
		"""Propogate input through the network."""
		src_emb = self.src_embedding(input_src)
		trg_emb = self.trg_embedding(input_trg)

		self.h0_encoder, self.c0_encoder = self.get_state(input_src)
		src_h, (src_h_t, src_c_t) = self.encoder(
			src_emb, (self.h0_encoder, self.c0_encoder)
		)

		if self.bidirectional:
			h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
			c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
		else:
			h_t = src_h_t[-1]
			c_t = src_c_t[-1]

		decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

		trg_h, (_, _) = self.decoder(
			trg_emb,
			(
				decoder_init_state.view(
					self.decoder.num_layers,
					decoder_init_state.size(0),
					decoder_init_state.size(1)
					),
				c_t.view(
					self.decoder.num_layers,
					c_t.size(0),
					c_t.size(1)
				)
			)
		)

		trg_h_reshape = trg_h.contiguous().view(
			trg_h.size(0) * trg_h.size(1),
			trg_h.size(2)
		)

		decoder_logit = self.decoder2vocab(trg_h_reshape)
		decoder_logit = decoder_logit.view(
			trg_h.size(0),
			trg_h.size(1),
			decoder_logit.size(1)
		)

		return decoder_logit

	def mle_loss(self, dec_logit, tgt):
		return self.criterion(
			dec_logit.contiguous().view(
				dec_logit.size(0) * dec_logit.size(1),
				-1
			),
			tgt.contiguous().view(-1)
		)

	def decode(self, logits):
		"""Return probability distribution over words."""
		logits_reshape = logits.view(-1, self.trg_vocab_size)
		word_probs = F.softmax(logits_reshape)
		word_probs = word_probs.view(
			logits.size()[0], logits.size()[1], logits.size()[2]
		)
		return word_probs


class SeqLM(nn.Module):
	def __init__(self, opts, dictionary):
		pass


if __name__ == '__main__':
	class Opts:
		bidirectional = True
		srcVocabSize = 100
		tgtVocabSize = 100
		embSize = 4
		hidSize = 3
		layerNum = 8

	opts = Opts()
	print opts

	from torch.autograd import Variable
	batch_tensor = torch.LongTensor([[9, 7, 10, 14, 0, 0], [96, 99, 32, 5, 9, 0]])
	mask_tensor = torch.FloatTensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
	batch_tensor_var = Variable(batch_tensor)
	mask_tensor_var = Variable(mask_tensor)

	emb_layer = nn.Embedding(opts.tgtVocabSize, opts.embSize)
	embs_var = emb_layer(batch_tensor_var).transpose(0, 1)

	rnnStack = StackedLSTM(opts)
	print rnnStack
	dirNum = 1
	if opts.bidirectional == True:
		dirNum = 2
	hids_0 = [(Variable(torch.zeros(2, opts.hidSize * dirNum)), Variable(torch.zeros(2, opts.hidSize  * dirNum))) for layer in range(opts.layerNum)]
	emb_t = embs_var.split(1)[0]
	# print emb_t.squeeze(0)
	# print hids_0
	output_l, new_hids = rnnStack(emb_t.squeeze(0), hids_0)
	print output_l
	print type(new_hids), len(new_hids)

	decoder = Decoder(opts)
	print decoder
	input4decoder = (batch_tensor_var, None, None)
	outputs = decoder(input4decoder, hids_0)
	print outputs.size()

