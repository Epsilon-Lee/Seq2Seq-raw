import torch
from seq2seq.Dict import Dict
from seq2seq.Dataset import Dataset, BilingualDataset
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
	trainSrcDataset = Dataset()
	trainSrcDataset.makeDataset('src', opts.trainSrcFilePath, srcDict, max_len)
	trainTgtDataset = Dataset()
	trainTgtDataset.makeDataset('tgt', opts.trainTgtFilePath, tgtDict, max_len)
	trainBilingualDataset = BilingualDataset(trainSrcDataset, trainTgtDataset)
	
	devSrcDataset = Dataset()
	devSrcDataset.makeDataset('src', opts.devSrcFilePath, srcDict, max_len)
	devTgtDataset = Dataset()
	devTgtDataset.makeDataset('tgt', opts.devTgtFilePath, tgtDict, max_len)
	devBilingualDataset = BilingualDataset(devSrcDataset, devTgtDataset)

	testSrcDataset = Dataset()
	testSrcDataset.makeDataset('src', opts.testSrcFilePath, srcDict, max_len)
	testTgtDataset = Dataset()
	testTgtDataset.makeDataset('tgt', opts.testTgtFilePath, tgtDict, max_len)
	testBilingualDataset = BilingualDataset(testSrcDataset, testTgtDataset)

	# test bilingual dataset
	testBilingualDataset.setBatchSize(8)
	# print('batchNum of test dataset: %d' % len(testBilingualDataset))
	# for idx in range(len(testBilingualDataset)):
	# 	srcBatch, tgtBatch = testBilingualDataset[idx]
	# 	print tgtBatch
	# 	break
	# testBilingualDataset.shuffle()
	# for idx in range(len(testBilingualDataset)):
	# 	srcBatch, tgtBatch = testBilingualDataset[idx]
	# 	print tgtBatch
	# 	break

	# testBilingualDataset.sortByTgtLength()
	# testBilingualDataset.shuffleBatch()
	# print testBilingualDataset[0]
	# print testBilingualDataset[1]
	print('Saving bilingual dataset...')
	torch.save(trainBilingualDataset, os.path.join(opts.path2save, 'trainDataset.pt'))
	torch.save(devBilingualDataset, os.path.join(opts.path2save, 'devDataset.pt'))
	torch.save(testBilingualDataset, os.path.join(opts.path2save, 'testDataset.pt'))
	print('Done.')

