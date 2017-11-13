import torch
import torch.optim as optim
from torch.autograd import Variable

from seq2seq.Models import Seq2Seq, Seq2Seq_MaximumEntropy
from seq2seq.Dict import Dict

import argparse
import time
import os
# from datetime import datatime

parser = argparse.ArgumentParser(description="train file options parser")

# Data path
parser.add_argument("-srcDictPath", type=str, default="IWSLT/de.30k.dict")
parser.add_argument("-tgtDictPath", type=str, default="IWSLT/en.30k.dict")
parser.add_argument("-trainDatasetPath", type=str, default="IWSLT/trainDataset.pt")
parser.add_argument("-devDatasetPath", type=str, default="IWSLT/devDataset.pt")
parser.add_argument("-testDatasetPath", type=str, default="IWSLT/testDataset.pt")

# Model options
parser.add_argument("-srcVocabSize", type=int, default=30000)
parser.add_argument("-tgtVocabSize", type=int, default=30000)
parser.add_argument("-embSize", type=int, default=256)
parser.add_argument("-hidSize", type=int, default=256)
parser.add_argument("-layerNum", type=int, default=2)
parser.add_argument("-packed", type=bool, default=True)

# Model options
parser.add_argument("-src_emb_size", type=int, default=256)
parser.add_argument("-tgt_emb_size", type=int, default=256)
parser.add_argument("-src_hid_dim", type=int, default=512)
parser.add_argument("-tgt_hid_dim", type=int, default=512)
parser.add_argument("-bidirectional", type=bool, default=True)
parser.add_argument("-src_num_layers", type=int, default=2)
parser.add_argument("-tgt_num_layers", type=int, default=1)
parser.add_argument("-batch_first", type=bool, default=True)
parser.add_argument("-dropout", type=float, default=0.)
parser.add_argument("-grad_threshold", type=float, default=5.)

# Training options
parser.add_argument("-max_epoch", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=1.)
parser.add_argument("-cuda", type=int, default=1)
parser.add_argument("-use_gpu", type=int, default=1)
parser.add_argument("-gpuid", type=int, default=2)

# Save model path
parser.add_argument("-save_path", type=str, default="../Models/Seq2Seq-raw_Models")
parser.add_argument("-model_prefix", type=str, default="seq2seq_me_bz64_mle")

# Logging options
parser.add_argument("-log_interval", type=int, default=16)
parser.add_argument("-visualize_interval", type=int, default=500)
parser.add_argument("-vis_num", type=int, default=8)
parser.add_argument("-log_file_path", type=str, default="../Logs/Seq2Seq-raw/Seq2SeqME_bz64_mle.txt")


opts = parser.parse_args()
print(opts)
# set gpu for usage
if opts.cuda:
	if torch.cuda.is_available():
		torch.cuda.set_device(opts.gpuid)
		# torch.backends.cudnn.enabled = False

# load dictionary
srcDictionary = Dict(opts, opts.srcDictPath)
tgtDictionary = Dict(opts, opts.tgtDictPath)

# load dataset
trainDataset = torch.load(opts.trainDatasetPath)
trainDataset.set_curriculum()
devDataset = torch.load(opts.devDatasetPath)
# testDataset = torch.load(opts.testDatasetPath)

trainDataset.set_batch_size(opts.batch_size)

# devDataset.setBatchSize(opts.batch_size)
# devDataset.sortByTgtLength()

# testDataset.setBatchSize(opts.batch_size)
# testDataset.sortByTgtLength()
print("Train set batch number: %d" % len(trainDataset))
print("Source vocabulary size: %d" % srcDictionary.size())
print("Target vocabulary size: %d" % tgtDictionary.size())
print("Model string prefix   : %s" % opts.model_prefix)
print("Log file path         : %s" % opts.log_file_path)
print("Run on gpu            : %d" % opts.gpuid)
# create model
# print(tgtDictionary.size())
seq2seq = Seq2Seq_MaximumEntropy(opts, srcDictionary, tgtDictionary)
seq2seq.init_weight()
if opts.cuda:
	seq2seq.cuda()
print("Model architecture:")
print(seq2seq)
# print(list(seq2seq.modules()))

# print("Model parameters:")
# print(list(seq2seq.parameters()))

# create optimizer
optimizer = optim.SGD(seq2seq.parameters(), lr=opts.lr)
# optimizer = optim.Adam(seq2seq.parameters(), lr=opts.lr)

loss_record = 0.
start_time = time.time()
loss_accum = 0
acc_count_total = 0
word_count_total = 0
# open logging file
if os.path.exists(opts.log_file_path):
	f_log = open(opts.log_file_path, 'a')
else:
	f_log = open(opts.log_file_path, 'w')
f_log.write('------------------------Start Experiment------------------------\n')
f_log.write(time.asctime(time.localtime(time.time())) + "\n")
f_log.write('----------------------------------------------------------------\n')
for epochIdx in range(1, opts.max_epoch + 1):
	# print("trainDataset size: %d batches" % len(trainDataset))
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
		if opts.cuda:
			src_batch[0] = Variable(src_batch[0].cuda())
			src_batch[1] = Variable(src_batch[1].cuda())
			# srcBatch[2] = Variable(torch.LongTensor(srcBatch[2]))
			tgt_batch[0] = Variable(tgt_batch[0].cuda())
			tgt_batch[1] = Variable(tgt_batch[1].cuda())
			# tgtBatch[2] = Variable(torch.LongTensor(tgtBatch[2]))

		seq2seq.zero_grad()
		logProbs = seq2seq(src_batch, tgt_batch)
		loss_batch = seq2seq.MLELoss(logProbs, tgt_batch)
		loss_batch.backward()

		# gradient clipping
		grad_norms = []
		params = seq2seq.parameters()
		for param in params:
			grad_norm_var = param.grad.norm()
			grad_norms.append(grad_norm_var.data[0])
			if grad_norm_var.data[0] > opts.grad_threshold:
				param.grad.data = param.grad.data / grad_norm_var.data
		grad_norm_avg = sum(grad_norms) / len(grad_norms)

		optimizer.step()

		# model predict: greedy
		maxProbs, pred_idxs = torch.max(logProbs.data, 2) # (bz x (seq_len - 1)) x vocab_size
		# print('pred_idxs size:', pred_idxs.size())
		gold_idxs = tgt_batch[0][:, 1:].contiguous().view(-1).data # (bz x (seq_len - 1))
		# print('gold_idxs size:', gold_idxs.size())

		acc_count = pred_idxs.eq(gold_idxs).masked_select(gold_idxs.ne(tgtDictionary.tgt_specials['<pad>'])).sum()

		loss_record = loss_batch.data[0]
		loss_accum += loss_record
		word_count_batch = sum(tgt_batch[2]) - len(tgt_batch[2])
		acc_count_total += acc_count
		word_count_total += word_count_batch
		# word_count_with_padding = tgtBatch[0].data.size(0) * tgtBatch[0].data.size(1)

		cur_acc = 1. * acc_count / word_count_batch
		if (idx + 1) % opts.log_interval == 0:
			# print(tgtBatch)
			print("Epoch %d Batch %d loss %f loss_avg %f acc: %f acc_avg: %f grad_norm_avg %f time elapsed: %f" 
					  % (epochIdx, idx + 1,
						loss_record / word_count_batch,
						# loss_record,
						# word_count_batch,
						# word_count_with_padding,
						loss_accum / word_count_total,
						acc_count * 1. / word_count_batch,
						acc_count_total * 1. / word_count_total,
						grad_norm_avg,
						time.time() - start_time))
			f_log.write("Epoch %d Batch %d loss %f loss_avg %f acc: %f acc_avg: %f time elapsed: %f\n" 
					  % (epochIdx, idx + 1, 
						loss_record / word_count_batch, 
						# loss_record, 
						# word_count_batch, 
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
				f_log.write('src     : %s\n' % src_symbol)
				f_log.write('tgt_pred: %s\n' % pred_symbol)
				f_log.write('tgt_gold: %s\n' % tgt_symbol)

		f_log.flush()

	# After each epoch, evaluate model in greedy mode


	# save model every epoch
	save_model_dict = {}
	model_state_dict = seq2seq.state_dict()
	save_model_dict['model_state_dict'] = model_state_dict
	save_model_dict['epoch'] = epochIdx
	model_string = opts.model_prefix + "-%d-acc_%.4f.pt" % (epochIdx, acc_count_total * 1. / word_count_total)
	torch.save(
		save_model_dict,
		os.path.join(opts.save_path, model_string)
	)

f_log.close()