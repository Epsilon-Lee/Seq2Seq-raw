import torch
from seq2seq.Dict import Dict
from seq2seq.Dataset import Dataset, BilingualDataset, SentencePairDataset
import argparse, os

parser = argparse.ArgumentParser(description="preprocess options")
parser.add_argument("-trainSrcFilePath", type=str, default="IWSLT/train.de.tok")
parser.add_argument("-trainTgtFilePath", type=str, default="IWSLT/train.en.tok")
parser.add_argument("-devSrcFilePath", type=str, default="IWSLT/dev.de.tok")
parser.add_argument("-devTgtFilePath", type=str, default="IWSLT/dev.en.tok")
parser.add_argument("-testSrcFilePath", type=str, default="IWSLT/test.de.tok")
parser.add_argument("-testTgtFilePath", type=str, default="IWSLT/test.en.tok")

parser.add_argument("-path2save", type=str, default="IWSLT")

parser.add_argument("-srcVocabSize", type=int, default=30000)
parser.add_argument("-tgtVocabSize", type=int, default=30000)

parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-max_len", type=int, default=50)

opts = parser.parse_args()

if __name__ == '__main__':
	# 1. create dictionary
	srcDict = Dict(opts)
	tgtDict = Dict(opts)

	srcDict.makeDict('src', opts.trainSrcFilePath)
	srcDict.pruneByVocabSize(opts.srcVocabSize)

	tgtDict.makeDict('tgt', opts.trainTgtFilePath)
	tgtDict.pruneByVocabSize(opts.tgtVocabSize)

	print("Before pruned src vocabulary size: %d" % len(srcDict.symbol_freq_lst))
	print("After pruned src vocabulary size: %d" % srcDict.size())
	print("Before pruned tgt vocabulary size: %d" % len(tgtDict.symbol_freq_lst))
	print("After pruned tgt vocabulary size: %d" % tgtDict.size())

	print("Saving dicts to %s ..." % opts.path2save)
	srcDict.save(os.path.join(opts.path2save, "de.30k.dict"))
	tgtDict.save(os.path.join(opts.path2save, "en.30k.dict"))
	print("Done.")
	# print tgtDict.symbol2idx
	# print tgtDict.symbol_freq_lst

	# 2. create train, dev, test 
	max_len = opts.max_len
	trainDataset = SentencePairDataset()
	trainDataset.make_dataset(
		'de2en-train', 
		opts.trainSrcFilePath,
		opts.trainTgtFilePath,
		srcDict,
		tgtDict,
		max_len
	)
	
	devDataset = SentencePairDataset()
	devDataset.make_dataset(
		'de2en-dev', 
		opts.devSrcFilePath,
		opts.devTgtFilePath,
		srcDict,
		tgtDict,
		None
	)

	testDataset = SentencePairDataset()
	testDataset.make_dataset(
		'de2en-test', 
		opts.testSrcFilePath,
		opts.testTgtFilePath,
		srcDict,
		tgtDict,
		None
	)

	# test SentencePairDataset
	batch_size = 8
	trainDataset.set_batch_size(batch_size)
	trainDataset.shuffle()
	trainDataset.set_curriculum()
	for idx in range(1):
		data_symbol, data_id, data_mask, data_lens = trainDataset[idx]
		for i in range(batch_size):
			src_sym, tgt_sym = data_symbol[i]
			src_id, tgt_id = data_id[0][i], data_id[1][i]
			src_mask, tgt_mask = data_mask[0][i], data_mask[1][i]
			src_len, tgt_len = data_lens[0][i], data_lens[1][i]
			print('src:')
			print(src_sym)
			print(src_id)
			print(src_mask)
			print(src_len)

			print('tgt:')
			print(tgt_sym)
			print(tgt_id)
			print(tgt_mask)
			print(tgt_len)

	# Save datasets
	print('Saving bilingual dataset...')
	torch.save(trainDataset, os.path.join(opts.path2save, 'trainDataset.pt'))
	torch.save(devDataset, os.path.join(opts.path2save, 'devDataset.pt'))
	torch.save(testDataset, os.path.join(opts.path2save, 'testDataset.pt'))
	print('Done.')

