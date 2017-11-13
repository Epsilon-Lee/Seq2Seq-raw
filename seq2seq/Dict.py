import torch

class Dict(object):

	src_specials = {'<pad>' : 0, '<unk>' : 1}
	tgt_specials = {'<pad>' : 0, '<unk>' : 1, '<s>' : 2, '</s>' : 3}

	def __init__(self, opts, dictPath=None):
		self.name = ''
		self.symbol2idx = {}
		self.idx2symbol = {}
		self.symbol_freq_lst = []
		if dictPath != None:
			savedDict = torch.load(dictPath)
			self.name = savedDict['name']
			self.symbol2idx = savedDict['symbol2idx']
			self.idx2symbol = savedDict['idx2symbol']
			self.symbol_freq_lst = savedDict['symbol_freq_lst']

	def makeDict(self, name, pathToCorpus):
		# tmp variable
		self.name = name # name is string from {'src', 'tgt'}
		symbol_freq_dict = {}
		with open(pathToCorpus) as f:
			lines = f.readlines()
			for line in lines:
				words = line.split()
				for word in words:
					if word in symbol_freq_dict:
						symbol_freq_dict[word] += 1
					else:
						symbol_freq_dict[word] = 1
		self.symbol_freq_lst = sorted(symbol_freq_dict.items(), key=lambda tp: tp[1], reverse=True)

	def loadDict(self, pathToDict):
		savedDict = torch.load(pathToDict)
		self.name = savedDict['name']
		self.symbol2idx = savedDict['symbol2idx']
		self.idx2symbol = savedDict['idx2symbol']
		self.symbol_freq_lst = savedDict['symbol_freq_lst']

	def addSpecials(self):
		if self.name == 'src':
			for special, idx in self.src_specials.iteritems():
				self.symbol2idx[special] = idx
				self.idx2symbol[idx] = special
		if self.name == 'tgt':
			for special, idx in self.tgt_specials.iteritems():
				self.symbol2idx[special] = idx
				self.idx2symbol[idx] = special

	def pruneByVocabSize(self, vocabSize):
		self.addSpecials()
		idx = len(self.symbol2idx)
		for i in range(vocabSize):
			self.symbol2idx[self.symbol_freq_lst[i][0]] = idx
			self.idx2symbol[idx] = self.symbol_freq_lst[i][0]
			idx += 1

	def pruneByFreq(self, freqThreshold):
		self.addSpecials()
		idx = len(self.symbol2idx)
		while self.symbol_freq_lst[idx][1] > freqThreshold:
			self.symbol2idx[self.symbol_freq_lst[idx - len(self.symbol2idx)][0]] = idx
			self.idx2symbol[idx] = self.symbol_freq_lst[idx - len(self.symbol2idx)][0]
			idx += 1

	def convertIdxSeq2SymbolSeq(self, idxSeq):
		symbolSeq = []
		for idx in idxSeq:
			symbolSeq.append(self.idx2symbol[idx])
		return symbolSeq

	def convertSymbolSeq2IdxSeq(self, symbolSeq):
		idxSeq = []
		for symbol in symbolSeq:
			if symbol in self.symbol2idx:
				idxSeq.append(self.symbol2idx[symbol])
			else:
				idxSeq.append(self.symbol2idx['<unk>'])
		return idxSeq

	def convert_id_lst_to_symbol_lst(self, id_lst):
		symbol_lst = []
		for id_seq in id_lst:
			first_eos_idx = id_seq.index(self.tgt_specials['</s>'])
			id_seq_pruned = id_seq[:first_eos_idx]
			symbol_seq = self.convertIdxSeq2SymbolSeq(id_seq_pruned)
			symbol_lst.append(symbol_seq)
		return symbol_lst

	def size(self):
		return len(self.symbol2idx)

	def save(self, path):
		savedDict = {}
		savedDict['name'] = self.name
		savedDict['symbol2idx'] = self.symbol2idx
		savedDict['idx2symbol'] = self.idx2symbol
		savedDict['symbol_freq_lst'] = self.symbol_freq_lst
		torch.save(savedDict, path)