import torch
import torch.optim as optim
from torch.autograd import Variable

from seq2seq.Models import Seq2Seq, Seq2SeqPyTorch
from seq2seq.Dict import Dict

import argparse
import time
import os

parser = argparse.ArgumentParser(description="train file options parser")

# Data path
parser.add_argument("-srcDictPath", type=str, default="IWSLT/de.30k.dict")
parser.add_argument("-tgtDictPath", type=str, default="IWSLT/en.30k.dict")
parser.add_argument("-trainDatasetPath", type=str, default="IWSLT/trainDataset.pt")
parser.add_argument("-devDatasetPath", type=str, default="IWSLT/devDataset.pt")
parser.add_argument("-testDatasetPath", type=str, default="IWSLT/testDataset.pt")

# Model options
parser.add_argument("-src_emb_dim", type=int, default=256)
parser.add_argument("-tgt_emb_dim", type=int, default=256)
parser.add_argument("-src_hid_dim", type=int, default=512)
parser.add_argument("-tgt_hid_dim", type=int, default=512)
parser.add_argument("-bidirectional", type=bool, default=True)
parser.add_argument("-src_num_layers", type=int, default=2)
parser.add_argument("-tgt_num_layers", type=int, default=1)
parser.add_argument("-batch_first", type=bool, default=True)
parser.add_argument("-dropout", type=float, default=0)
# parser.add_argument("-packed", type=bool, default=True)

# Training options
parser.add_argument("-max_epoch", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-use_gpu", type=int, default=1)
parser.add_argument("-gpuid", type=int, default=1)

# Save model path
parser.add_argument("-save_path", type=str, default="../Models/Seq2Seq-raw_Models")

# Logging options
parser.add_argument("-log_interval", type=int, default=50)
parser.add_argument("-visualize_interval", type=int, default=500)
parser.add_argument("-vis_num", type=int, default=8)

opts = parser.parse_args()

# set gpu for usage
if opts.use_gpu:
	if torch.cuda.is_available():
		torch.cuda.set_device(opts.gpuid)
		# torch.backends.cudnn.enabled = False

# load dictionary
srcDictionary = Dict(opts, opts.srcDictPath)
tgtDictionary = Dict(opts, opts.tgtDictPath)

# load dataset
trainDataset = torch.load(opts.trainDatasetPath)

trainDataset.set_batch_size(opts.batch_size)
# trainDataset.shuffle()

# create model
print(srcDictionary.size())
print(tgtDictionary.size())
print(opts)
seq2seq = Seq2SeqPyTorch(
	opts.src_emb_dim,
	opts.tgt_emb_dim,
	srcDictionary.size(),
	tgtDictionary.size(),
	opts.src_hid_dim,
	opts.tgt_hid_dim,
	opts.batch_size,
	srcDictionary.src_specials['<pad>'],
	tgtDictionary.tgt_specials['<pad>'],
	bidirectional=opts.bidirectional,
	nlayers=opts.src_num_layers,
	nlayers_trg=opts.tgt_num_layers,
	dropout=opts.dropout
)
seq2seq.init_weights()
if opts.use_gpu:
	seq2seq.cuda()
print("Model architecture:")
print(seq2seq)
# print(list(seq2seq.modules()))

# print("Model parameters:")
# print(list(seq2seq.parameters()))

# create optimizer
# sgdOptimizer = optim.SGD(seq2seq.parameters(), lr=opts.lr)
sgdOptimizer = optim.Adam(seq2seq.parameters(), lr=opts.lr)

loss_record = 0.
start_time = time.time()
loss_accum = 0
acc_count_total = 0
word_count_total = 0
for epochIdx in range(1, opts.max_epoch + 1):
	trainDataset.shuffle()
	for idx in range(len(trainDataset)):
		data_symbol, data_id, data_mask, data_lens = trainDataset[idx]
		bz_local = len(data_lens[0])
		src_id = data_id[0]
		tgt_id = data_id[1]
		src_mask = data_mask[0]
		tgt_mask = data_mask[1]
		src_lens = data_lens[0]
		tgt_lens = data_lens[1]
		src_batch = [src_id, src_mask, src_lens]
		tgt_batch = [tgt_id, tgt_mask, tgt_lens]
		if opts.use_gpu:
			src_batch[0] = Variable(src_batch[0].cuda())
			src_batch[1] = Variable(src_batch[1].cuda())
			# srcBatch[2] = Variable(torch.LongTensor(srcBatch[2]))
			tgt_batch[0] = Variable(tgt_batch[0].cuda())
			tgt_batch[1] = Variable(tgt_batch[1].cuda())
			# tgtBatch[2] = Variable(torch.LongTensor(tgtBatch[2]))

		seq2seq.zero_grad()
		logProbs = seq2seq(src_batch[0], tgt_batch[0][:, :-1])
		# print(logProbs.size())
		loss_batch = seq2seq.mle_loss(logProbs, tgt_batch[0][:, 1:])
		loss_batch.backward()
		sgdOptimizer.step()

		# model predict: greedy
		maxProbs, pred_idxs = torch.max(logProbs.data, 2) # (bz x (seq_len - 1) ) x vocab_size
		gold_idxs = tgt_batch[0][:, 1:].contiguous().view(-1).data # (bz x (seq_len - 1))
		src_idxs = src_batch[0].data

		# print(pred_idxs.size())
		# print(gold_idxs.size())

		acc_count = pred_idxs.eq(gold_idxs).masked_select(gold_idxs.ne(tgtDictionary.tgt_specials['<pad>'])).sum()
		acc_count_total += acc_count
		loss_record = loss_batch.data[0]
		loss_accum += loss_record
		word_count_batch = sum(tgt_batch[2]) - len(tgt_batch[2])
		word_count_total += word_count_batch
		# word_count_with_padding = tgtBatch[0].data.size(0) * tgtBatch[0].data.size(1)

		# cur_acc = 1. * acc_count / word_count_batch
		
		if (idx + 1) % opts.log_interval == 0:
			# print(tgtBatch)
			print("Epoch %d Batch %d loss %f loss_avg %f acc: %f acc_avg: %f time elapsed: %f" 
					  % (epochIdx, idx + 1, 
						loss_record / word_count_batch, 
						# word_count_with_padding, 
						loss_accum / word_count_total, 
						acc_count * 1. / word_count_batch, 
						acc_count_total * 1. / word_count_total,
						time.time() - start_time))

		if (idx + 1) % opts.visualize_interval == 0:
			pred_idxs = pred_idxs.contiguous().view(bz_local, -1) # bz x (seq_len - 1)
			# print(pred_idxs.size())
			pred_idxs = pred_idxs.tolist()
			print('--------------------------------------------------------------')
			for i in xrange(opts.vis_num):
				src_symbol, tgt_symbol = data_symbol[i]
				pred_idx = pred_idxs[i]
				# pred_idx = pred_idx[:pred_idx.index(tgtDictionary.tgt_specials['</s>'])]
				pred_symbol = " ".join(tgtDictionary.convertIdxSeq2SymbolSeq(pred_idx))
				print('src     : %s' % src_symbol)
				print('tgt_pred: %s' % pred_symbol)
				print('tgt_gold: %s' % tgt_symbol)
				print

	# After each epoch, evaluate model in greedy mode
	print('----------Evaluating the model on dev dataset----------')
	f_log.write('----------Evaluating the model on dev dataset----------\n')
	print('In greedy decoding......')
	f_log.write('In greedy decoding......\n')
	cand_lst = []
	ref_lst = []
	for idx in tqdm(xrange(len(devDataset))):
		dev_data_symbol, dev_data_id, dev_data_mask, dev_data_lens = devDataset[idx]
		src_id = Variable(dev_data_id[0])
		_ , pred = seq2seq.greedy_search(src_id, opts.maximum_decode_length)
		pred_lst = pred.tolist() # bz x max_dec_len
		cand_lst_batch = tgtDictionary.convert_id_lst_to_symbol_lst(pred_lst)
		cand_lst.extend(cand_lst_batch)
		ref_lst_batch = [symbol_tuple[1] for symbol_tuple in dev_data_symbol]
		ref_lst.extend(ref_lst_batch)
		# print(len(ref_lst_batch), len(cand_lst_batch))
	bleu_1_to_4, bleu, bp, hyp_ref_len, ratio = bleu_calculator.calc_bleu(cand_lst, [ref_lst])
	f_log.write("%s %s, %s, %s, %s, bp=%s, ratio=%s\n"
		% (format(bleu, "2.2%"),
			format(bleu_1_to_4[0], "2.2%"),
			format(bleu_1_to_4[1], "2.2%"),
			format(bleu_1_to_4[2], "2.2%"),
			format(bleu_1_to_4[3], "2.2%"),
			format(bp, "0.4f"),
			format(ratio, "0.4f"))
	)
	print("%s %s, %s, %s, %s, bp=%s, ratio=%s"
		% (format(bleu, "2.2%"),
			format(bleu_1_to_4[0], "2.2%"),
			format(bleu_1_to_4[1], "2.2%"),
			format(bleu_1_to_4[2], "2.2%"),
			format(bleu_1_to_4[3], "2.2%"),
			format(bp, "0.4f"),
			format(ratio, "0.4f"))
	)
	if bleu > max_bleu:
		max_bleu = bleu

	# save model every epoch
	# saveModelDict = {}
	# model_state_dict = seq2seq.state_dict()
	# saveModelDict['model_state_dict'] = model_state_dict
	# saveModelDict['epoch'] = epochIdx
	# torch.save(saveModelDict, os.path.join(opts.save_path, str(epochIdx) + str(acc_count * 1./ word_count_batch)))

