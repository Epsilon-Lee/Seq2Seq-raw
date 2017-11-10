import torch
from torch.autograd import Variable
import sys
import os
import argparse

sys.path.append('../')
# print sys.path

from seq2seq.Models import Seq2Seq_MaximumEntropy
from seq2seq.Dict import Dict

model_path = "../../Models/Seq2Seq-raw_Models/seq2seq_me_bz32_adam-60-acc_0.6018.pt"

if os.path.exists(model_path):
	print('Model exists!')

parser = argparse.ArgumentParser("debug_search's argument parser")

parser.add_argument("-devDatasetPath", type=str, default="../IWSLT/devDataset.pt")

parser.add_argument("-srcDictPath", type=str, default="../IWSLT/de.30k.dict")
parser.add_argument("-tgtDictPath", type=str, default="../IWSLT/en.30k.dict")

parser.add_argument("-gpuid", type=int, default=2)
parser.add_argument("-use_gpu", type=int, default=1)

parser.add_argument("-src_emb_size", type=int, default=256)
parser.add_argument("-tgt_emb_size", type=int, default=256)
parser.add_argument("-src_hid_dim", type=int, default=512)
parser.add_argument("-tgt_hid_dim", type=int, default=512)
parser.add_argument("-bidirectional", type=bool, default=True)
parser.add_argument("-src_num_layers", type=int, default=2)
parser.add_argument("-tgt_num_layers", type=int, default=1)
parser.add_argument("-batch_first", type=bool, default=True)
parser.add_argument("-dropout", type=float, default=0)

opts = parser.parse_args()
src_dict = Dict(opts, opts.srcDictPath)
tgt_dict = Dict(opts, opts.tgtDictPath)

seq2seq_me = Seq2Seq_MaximumEntropy(opts, src_dict, tgt_dict)
# seq2seq_me.load_state_dict(torch.load(model_path)['model_state_dict'])
print(seq2seq_me)

dev_dataset = torch.load(opts.devDatasetPath)
dev_dataset.set_batch_size(4)
data_symbol, data_id, data_mask, data_lens = dev_dataset[0]
print(type(data_symbol)) # tuple of str
print(data_symbol)
print(data_id[0])
print(data_id[1])

src_id = data_id[0]
seq2seq_me.greedy_search()