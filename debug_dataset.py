import torch
from seq2seq.Dict import Dict

train_dataset = torch.load('IWSLT/trainDataset.pt')
de_dict = Dict(None, 'IWSLT/de.30k.dict')
en_dict = Dict(None, 'IWSLT/en.30k.dict')

print(type(train_dataset))
train_dataset.setBatchSize(8)
train_dataset.setCurriculum()
train_dataset.setBatchMode(False)

train_dataset.setBatchMode(True)
train_dataset.sortByTgtLength()
train_dataset.shuffleBatch()
for i in xrange(len(train_dataset)):
	src, tgt = train_dataset[i]
	src_id_lst = src[0].tolist()
	tgt_id_lst = tgt[0].tolist()
	for src_id_seq, tgt_id_seq in zip(src_id_lst, tgt_id_lst):
		print('Source sentence: %s' % " ".join(de_dict.convertIdxSeq2SymbolSeq(src_id_seq)))
		print('Target sentence: %s' % " ".join(en_dict.convertIdxSeq2SymbolSeq(tgt_id_seq)))
		print
	if i == 10:
		break
