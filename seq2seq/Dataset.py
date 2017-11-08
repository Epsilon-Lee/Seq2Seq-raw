import torch
import random
import math

class SentencePairDataset(object):

	MAX_LEN = 1000

	def __init__(self):
		self.name = '' # de2en
		self.data_symbol = []
		self.data_id = []
		self.batch_num = 0
		self.batch_size = 0
		self.cuda = False

	def make_dataset(
		self,
		name,
		path_to_a,
		path_to_b,
		dict_a,
		dict_b,
		max_len=None
	):
		if max_len is None:
			max_len = self.MAX_LEN
		lines_a = []
		lines_b = []
		with open(path_to_a, 'r') as f_a:
			lines_a = f_a.readlines()
		with open(path_to_b, 'r') as f_b:
			lines_b = f_b.readlines()
		for line_a, line_b in zip(lines_a, lines_b):
			# line_a, line_b should not be '' string
			line_a, line_b = line_a.strip(), line_b.strip()
			self.data_symbol.append((line_a, line_b))
			# cut off by max_len
			line_a_words, line_b_words = line_a.split(), line_b.split()
			line_a_words = line_a_words[:max_len]
			line_b_words = line_b_words[:max_len]
			# convert to ids
			line_a_id = dict_a.convertSymbolSeq2IdxSeq(line_a_words)
			line_b_id = dict_b.convertSymbolSeq2IdxSeq(['<s>'] + line_b_words + ['</s>'])
			self.data_id.append((line_a_id, line_b_id))

		srclen = [len(tup[0]) for tup in self.data_id]
		symbol_id_srclen = zip(self.data_symbol, self.data_id, srclen)
		symbol_id_srclen_sorted = sorted(symbol_id_srclen, key=lambda tup: tup[2])
		self.data_symbol, self.data_id, _ = zip(*symbol_id_srclen_sorted)

		# # test
		# for symbol, ids in zip(self.data_symbol, self.data_id):
		# 	print symbol
		# 	print ids
		# 	print len(ids[0])
		# 	print 

	def __len__(self):
		assert self.batch_size is not 0, "batch_size is not set"
		return self.batch_num

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.batch_num = int(math.ceil(len(self.data_id) * 1. / self.batch_size))
		self.order = torch.randperm(self.batch_num).tolist()

	def set_curriculum(self):
		self.order = [idx for idx in xrange(self.batch_size)]

	def shuffle(self):
		self.order = torch.randperm(self.batch_num).tolist()

	def _batchify(self, data_symbol_batch, data_id_batch):
		bz = len(data_symbol_batch)
		src_lens_batch = [len(data_id[0]) for data_id in data_id_batch]
		tgt_lens_batch = [len(data_id[1]) for data_id in data_id_batch]
		# sorted by src
		symbol_id_srclen_tgtlen = zip(data_symbol_batch, data_id_batch, src_lens_batch, tgt_lens_batch)
		symbol_id_srclen_tgtlen_sorted = sorted(symbol_id_srclen_tgtlen, key=lambda tup: tup[2], reverse=True)
		data_symbol_batch, data_id_batch, src_lens_batch, tgt_lens_batch = zip(*symbol_id_srclen_tgtlen_sorted)

		# test
		# for i in xrange(bz):
		# 	print data_symbol_batch[i]
		# 	print data_id_batch[i]
		# 	print src_lens_batch[i]
		# 	print tgt_lens_batch[i]
		# 	print

		max_len_src = max(src_lens_batch)
		max_len_tgt = max(tgt_lens_batch)

		src_id_batch, tgt_id_batch = zip(*data_id_batch)
		
		# padding src, tgt
		src_id_tensor = torch.LongTensor(bz, max_len_src).fill_(0)
		src_mask_tensor = torch.FloatTensor(bz, max_len_src).fill_(0)
		tgt_id_tensor = torch.LongTensor(bz, max_len_tgt).fill_(0)
		tgt_mask_tensor = torch.LongTensor(bz, max_len_tgt).fill_(0)
		for i in xrange(bz):
			src_id = torch.LongTensor(src_id_batch[i])
			src_mask = torch.FloatTensor(src_lens_batch[i]).fill_(1)
			tgt_id = torch.LongTensor(tgt_id_batch[i])
			tgt_mask = torch.FloatTensor(tgt_lens_batch[i]).fill_(1)
			src_id_tensor[i].narrow(0, 0, src_lens_batch[i]).copy_(src_id)
			src_mask_tensor[i].narrow(0, 0, src_lens_batch[i]).copy_(src_mask)
			tgt_id_tensor[i].narrow(0, 0, tgt_lens_batch[i]).copy_(tgt_id)
			tgt_mask_tensor[i].narrow(0, 0, tgt_lens_batch[i]).copy_(tgt_mask)

		data_id_batch = (src_id_tensor, tgt_id_tensor)
		data_mask_batch = (src_mask_tensor, tgt_mask_tensor)
		data_lens_batch = (src_lens_batch, tgt_lens_batch)
		return data_symbol_batch, data_id_batch, data_mask_batch, data_lens_batch

	def __getitem__(self, idx):
		idx = self.order[idx]
		bz = self.batch_size
		data_symbol_batch = self.data_symbol[idx * bz : (idx + 1) * bz]
		data_id_batch = self.data_id[idx * bz : (idx + 1) * bz]
		data_symbol_batch, data_id_batch, data_mask_batch, data_lens_batch = self._batchify(data_symbol_batch, data_id_batch)
		return data_symbol_batch, data_id_batch, data_mask_batch, data_lens_batch


class Dataset(object):
	def __init__(self):
		self.name = ''
		self.dataSymbol = [] # self.dataSymbol and self.dataIdx are corresponded
		self.dataIdx = []
		self.batchNum = 0
		self.batchSize = 0

	def __len__(self):
		return self.batchNum

	def makeDataset(self, name, pathToCorpus, dictionary, max_len):
		self.name = name
		with open(pathToCorpus, 'r') as f:
			lines = f.readlines()
			for line in lines:
				words = line.split()
				words = words[:max_len] # cut by max_len
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

	def setCurriculum(self):
		self.order = [idx for idx in range(self.batchNum)]

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
		src_tgt = sorted(src_tgt, key=lambda tp: len(tp[3]))
		self.srcDataset.dataSymbol, self.srcDataset.dataIdx, self.tgtDataset.dataSymbol, self.tgtDataset.dataIdx = zip(*src_tgt)
		# for idxSeq in self.srcDataset.dataIdx:
		# 	print(len(idxSeq))

	def shuffleBatch(self):
		self.order = torch.randperm(self.batchNum)

	
