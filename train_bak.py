import torch
import torch.optim as optim
from torch.autograd import Variable

from seq2seq.Models import Seq2Seq
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
parser.add_argument("-srcVocabSize", type=int, default=30000)
parser.add_argument("-tgtVocabSize", type=int, default=30000)
parser.add_argument("-embSize", type=int, default=256)
parser.add_argument("-hidSize", type=int, default=256)
parser.add_argument("-bidirectional", type=bool, default=True)
parser.add_argument("-layerNum", type=int, default=2)
parser.add_argument("-packed", type=bool, default=True)

# Training options
parser.add_argument("-max_epoch", type=int, default=15)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-cuda", type=int, default=1)
parser.add_argument("-gpuid", type=int, default=3)

# Save model path
parser.add_argument("-save_path", type=str, default="../Models/Seq2Seq-raw_Models")

# Logging options
parser.add_argument("-log_interval", type=int, default=50)

opts = parser.parse_args()

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
devDataset = torch.load(opts.devDatasetPath)
testDataset = torch.load(opts.testDatasetPath)

trainDataset.setBatchSize(opts.batch_size)
trainDataset.sortByTgtLength()

devDataset.setBatchSize(opts.batch_size)
devDataset.sortByTgtLength()

testDataset.setBatchSize(opts.batch_size)
testDataset.sortByTgtLength()

# create model
print(tgtDictionary.size())
seq2seq = Seq2Seq(opts, srcDictionary, tgtDictionary)
seq2seq.init_weight()
if opts.cuda:
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
word_count_total = 0
for epochIdx in range(1, opts.max_epoch + 1):
	trainDataset.setBatchMode(True)
	trainDataset.shuffleBatch()
	for idx in range(len(trainDataset)):
		srcBatch, tgtBatch = trainDataset[idx]
		srcBatch = list(srcBatch)
		tgtBatch = list(tgtBatch)
		if opts.cuda:
			srcBatch[0] = Variable(srcBatch[0].cuda())
			srcBatch[1] = Variable(srcBatch[1].cuda())
			# srcBatch[2] = Variable(torch.LongTensor(srcBatch[2]))
			tgtBatch[0] = Variable(tgtBatch[0].cuda())
			tgtBatch[1] = Variable(tgtBatch[1].cuda())
			# tgtBatch[2] = Variable(torch.LongTensor(tgtBatch[2]))

		seq2seq.zero_grad()
		logProbs = seq2seq(srcBatch, tgtBatch, opts)
		loss_batch = seq2seq.MLELoss(logProbs, tgtBatch)
		loss_batch.backward()
		sgdOptimizer.step()

		# model predict: greedy
		maxProbs, pred_idxs = torch.max(logProbs.data, 1) # ((seq_len - 1) x bz) x vocab_size
		gold_idxs = tgtBatch[0].t()[1:].contiguous().view(-1).data # ((seq_len - 1) x bz)

		acc_count = pred_idxs.eq(gold_idxs).masked_select(gold_idxs.ne(tgtDictionary.tgt_specials['<pad>'])).sum()

		loss_record = loss_batch.data[0]
		loss_accum += loss_record
		word_count_batch = sum(tgtBatch[2]) - len(tgtBatch[2])
		word_count_total += word_count_batch
		# word_count_with_padding = tgtBatch[0].data.size(0) * tgtBatch[0].data.size(1)

		cur_acc = 1. * acc_count / word_count_batch
		if (idx + 1) % opts.log_interval == 0:
			# print(tgtBatch)
			print("Epoch %d Batch %d loss %f  (%f / %d) loss_avg %f acc: %f time elapsed: %f" 
					  % (epochIdx, idx + 1, 
						loss_record / word_count_batch, 
						loss_record, 
						word_count_batch, 
						# word_count_with_padding, 
						loss_accum / word_count_total, 
						acc_count * 1. / word_count_batch, 
						time.time() - start_time))
	
	# save model every epoch
	saveModelDict = {}
	model_state_dict = seq2seq.state_dict()
	saveModelDict['model_state_dict'] = model_state_dict
	saveModelDict['epoch'] = epochIdx
	torch.save(saveModelDict, os.path.join(opts.save_path, str(epochIdx) + str(acc_count * 1./ word_count_batch)))

