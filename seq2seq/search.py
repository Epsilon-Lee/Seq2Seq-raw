import torch
from torch.autograd import Variable

def greedy_search(
	model,
	input,
	input_mask,
	input_lengths,
	dec_dict,
	max_decode_len,
	USE_GPU
):
	"""Utility function to execute greedy search decoding
	for a given batch input

	Args
	----------
	model          : Seq2Seq / BahdahnauAttentionSeq2Seq / GlobalAttentionSeq2Seq
		Model objects which has four sub-modules: 1). encoder, 2). bridge,
		3). decoder, 4). generator; so that the search function can call
		those sub-modules via `.` operation.

	input            : N x enc_L
	input_mask       : N x enc_L
	input_lengths    : list len()=>N
	dec_dict         : Dict
		provides id number of specials '<s>' and '</s>'
	max_decode_len   : int
	USE_GPU          : int (0 or 1)

	Return
	----------
	pred_ids         : N x max_decode_len
	att              : N x enc_L x max_decode_len / []
	total_log_probs  : N
	"""
	N = input.size(0)
	enc_L = input.size(1)
	BOS = dec_dict.tgt_specials['<s>']
	EOS = dec_dict.tgt_specials['</s>']

	# set the model to evaluatioin mode, to save memory
	model.eval()

	enc_hids, enc_last = model.encoder(
		input,
		input_mask,
		input_lengths
	)

	dec_init = model.bridge(enc_last)
	dec_prev = dec_init
	if USE_GPU:
		dec_input_t = Variable(
			torch.LongTensor(N, 2).fill_(BOS)
		).cuda() # in (N, 2), `2` is a trick, since in model.decoder(...), 
		# the (N, 2) will be sliced to (N, 1)
	else:
		dec_input_t = Variable(
			torch.LongTensor(N, 2).fill_(BOS)
		)
		
	pred_ids = []
	att = []
	if USE_GPU:
		total_log_probs = torch.FloatTensor(N).fill_(0).cuda()
		mask = torch.FloatTensor(N).fill_(1).cuda()
	else:
		total_log_probs = torch.FloatTensor(N).fill_(0)
		mask = torch.FloatTensor(N).fill_(1)
	
	for t_step in xrange(max_decode_len):

		if model.name == 'Seq2Seq':
			dec_hids_curr, dec_prev = model.decoder(
				dec_input_t,
				dec_prev
			) # N x 1 x dec_H
		else:
			dec_hids_curr, att_curr, dec_prev = model.decoder(
				dec_input_t,
				dec_prev,
				enc_hids
			) # N x 1 x dec_H, N x enc_L x 1, tuple / dec_nL x N x dec_H
			att.append(att_curr.squeeze(2))
		pred_dstrs_t = model.generator(dec_hids_curr).squeeze(1) # N x 1 x vocab_size => N x vocab_size
		log_probs, ids = torch.max(pred_dstrs_t, dim=1) # N: torch.FloatTensor; N: torch.LongTensor
		dec_input_t = torch.stack([ids, ids], dim=1) # N x 2: Variable(torch.LongTensor)

		total_log_probs += log_probs.data * mask
		pred_ids.append(ids)
		
		# build mask log_probs
		for i in range(N):
			if ids.data[i] == EOS:
				mask[i] = 0

	pred_ids = torch.stack(pred_ids, dim=0).t() # N x max_dec_len
	if len(att) != 0:
		att = torch.stack(att, dim=2) # N x enc_L x max_dec_len

	# set the model back to train mode
	model.train()

	return pred_ids, att, total_log_probs