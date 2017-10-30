import torch
import torch.optim as optim
from torch.autograd import Variable

from seq2seq.Models import Seq2Seq
from seq2seq.Dict import Dict

import argparse

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
parser.add_argument("-lr", type=float, default=1.0)
parser.add_argument("-cuda", type=int, default=1)
parser.add_argument("-gpuid", type=int, default=0)

# Logging options
parser.add_argument("-log_interval", type=int, default=50)

opts = parser.parse_args()

# set gpu for usage
if opts.cuda:
	if torch.cuda.is_available():
		torch.cuda.set_device(opts.gpuid)

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
seq2seq = Seq2Seq(opts, tgtDictionary)
if opts.cuda:
	seq2seq.cuda()
print("Model architecture:")
print(seq2seq)

# print("Model parameters:")
# print(list(seq2seq.parameters()))

# create optimizer
sgdOptimizer = optim.SGD(seq2seq.parameters(), lr=opts.lr)

loss_record = 0.
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

		loss_record = loss_batch.data[0]

		if (idx + 1) % opts.log_interval == 0:
			print("Batch %d loss %f" % (idx + 1, loss_record / opts.batch_size))
	break
