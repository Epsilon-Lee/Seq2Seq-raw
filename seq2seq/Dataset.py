import torch
import random


class Dataset(object):
	def __init__(self):
		self.name = ''
		self.dataSymbol = [] # self.dataSymbol and self.dataIdx are corresponded
		self.dataIdx = []
		self.batchNum = 0
		self.batchSize = 0

	def __len__(self):
		return self.batchNum

	def makeDataset(self, name, pathToCorpus, dictionary):
		self.name = name
		with open(pathToCorpus, 'r') as f:
			lines = f.readlines()
			for line in lines:
				words = line.split()
				if name == 'tgt':
					self.dataSymbol.append(['<s>'] + words + ['</s>'])
				else:
					self.dataSymbol.append(words)
		for symbolSeq in self.dataSymbol:
			idxSeq = dictionary.convertSymbolSeq2IdxSeq(symbolSeq)
			self.dataIdx.append(idxSeq)

	def setBatchSize(self, batchSize):
		assert batchSize != 0, "batchSize should not be 0"
		self.batchSize = batchSize
		self.batchNum = int(len(self.dataIdx) * 1. // batchSize + 1)

	def _batchify(self, batchIdx):
		batchLengths = [len(idxSeq) for idxSeq in batchIdx]

		# sort in reverse mode
		batch_lengths = zip(batchLengths, batchIdx)
		batch_lengths_sorted = sorted(batch_lengths, key=lambda tup: tup[0], reverse=True)
		batchLengths, batchIdx = zip(*batch_lengths_sorted)

		maxLength = max(batchLengths)
		currentBatchSize = len(batchIdx)

		batchIdxPt = torch.LongTensor(currentBatchSize, maxLength).zero_() # fill_(0)
		batchMaskPt = torch.FloatTensor(currentBatchSize, maxLength).zero_()

		for i, idxSeq in enumerate(batchIdx):
			idxSeqPt = torch.LongTensor(idxSeq)
			maskSeqPt = torch.FloatTensor(batchLengths[i]).fill_(1)
			batchIdxPt[i].narrow(0, 0, batchLengths[i]).copy_(idxSeqPt)
			batchMaskPt[i].narrow(0, 0, batchLengths[i]).copy_(maskSeqPt)

		return batchIdxPt, batchMaskPt, batchLengths

	def __getitem__(self, idx):
		assert idx < self.batchNum, 'idx out of range!'
		batchIdx = self.dataIdx[idx * self.batchSize : (idx + 1) * self.batchSize]
		batchIdxPt, batchMaskPt, batchLengths = self._batchify(batchIdx)
		return (batchIdxPt, batchMaskPt, batchLengths)


class BilingualDataset(object):
	def __init__(self, srcDataset, tgtDataset):
		self.srcDataset = srcDataset
		self.tgtDataset = tgtDataset
		self.batchNum = self.srcDataset.batchNum
		self.batchSize = self.srcDataset.batchSize

	def __len__(self):
		return self.batchNum

	def setBatchSize(self, batchSize):
		self.srcDataset.setBatchSize(batchSize)
		self.tgtDataset.setBatchSize(batchSize)
		self.batchNum = self.srcDataset.batchNum
		self.batchSize = batchSize

	def setBatchMode(self, mode):
		self.batchMode = mode

	def __getitem__(self, idx):
		if self.batchMode:
			realIdx = int(self.order[idx])
			srcBatch = self.srcDataset[realIdx]
			tgtBatch = self.tgtDataset[realIdx]
			return srcBatch, tgtBatch
		else:
			srcBatch = self.srcDataset[idx]
			tgtBatch = self.tgtDataset[idx]
			return srcBatch, tgtBatch

	def shuffle(self):
		src_tgt = zip(self.srcDataset.dataSymbol, self.srcDataset.dataIdx, self.tgtDataset.dataSymbol, self.tgtDataset.dataIdx)
		random.shuffle(src_tgt)
		self.srcDataset.dataSymbol, self.srcDataset.dataIdx, self.tgtDataset.dataSymbol, self.tgtDataset.dataIdx = zip(*src_tgt)

	def sortByTgtLength(self):
		src_tgt = zip(self.srcDataset.dataSymbol, self.srcDataset.dataIdx, \
					self.tgtDataset.dataSymbol, self.tgtDataset.dataIdx)
		# print type(src_tgt)
		sorted(src_tgt, key=lambda tp: len(tp[3]))
		self.srcDataset.dataSymbol, self.srcDataset.dataIdx, self.tgtDataset.dataSymbol, self.tgtDataset.dataIdx = zip(*src_tgt)
		# for idxSeq in self.srcDataset.dataIdx:
		# 	print(len(idxSeq))

	def shuffleBatch(self):
		self.order = torch.randperm(self.batchNum)

	def curriculum(self):
		pass
