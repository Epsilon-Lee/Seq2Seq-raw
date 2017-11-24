import sys
sys.path.append('/disk2/glli/Workspace/NMT/Seq2Seq-raw/')

import json
import torch
import torch.cuda as cuda
from torch.autograd import Variable
from seq2seq.seq2seq_models import BahdahnauAttentionSeq2Seq
from seq2seq.Dict import Dict
from seq2seq.search import greedy_search
from seq2seq.bleucal import BleuCalculator
from tqdm import tqdm

saved_model_path = '../../Models/Seq2Seq-raw_Models/modular_batchid_25.pt'
model_state_dict = torch.load(saved_model_path)

# # print param name and value
# for name, param in model_state_dict.iteritems():
# 	print(name)
# 	print(param)

# create config object
config_file_path = '../config_de_en_vanilla_iwslt.json'
config = json.load(open(config_file_path))

# load src_dict and tgt_dict
src_dict = Dict(None, '../' + config['data']['src_dict_path'])
tgt_dict = Dict(None, '../' + config['data']['tgt_dict_path'])

# load dev and test dataset
dev_data   = torch.load('../' + config['data']['dev_path'])
test_data  = torch.load('../' + config['data']['test_path'])

dev_data.set_batch_size(config['training']['dev_batch_size'])
test_data.set_batch_size(config['training']['test_batch_size'])

# create bleu calculator
bleu_calculator = BleuCalculator(None, None)

# create BahdahnauAttentionSeq2Seq model
model = BahdahnauAttentionSeq2Seq(
	src_dict,
	src_dict.src_specials['<pad>'],
	config['model']['encoder']['emb_size'],
	config['model']['encoder']['hid_size'],
	config['model']['encoder']['bidirectional'],
	config['model']['encoder']['rnn_cell_type'],
	config['model']['encoder']['is_packed'],
	config['model']['encoder']['batch_first'],
	config['model']['encoder']['num_layers'],
	config['model']['encoder']['dropout'],
	tgt_dict,
	tgt_dict.tgt_specials['<pad>'],
	config['model']['decoder']['emb_size'],
	config['model']['decoder']['hid_size'],
	config['model']['decoder']['rnn_cell_type'],
	config['model']['decoder']['num_layers'],
	config['model']['decoder']['dropout'],
	config['model']['generator']['dim_lst'],
	config['model']['generator']['num_layers']
)

model.load_state_dict(model_state_dict)

# print(model)

USE_GPU = 1
if USE_GPU:
	model.cuda()
	if cuda.is_available():
		cuda.set_device(0) # use gpu 0 for unit test
else:
	model.cpu()

# greedy decoding
cand_lst = []
gold_lst = []
for devBatchIdx in tqdm(xrange(len(dev_data))):

	data_symbol, data_id, data_mask, data_lengths = dev_data[devBatchIdx]
	src_id, tgt_id = data_id
	src_mask, tgt_mask = data_mask
	src_lengths, tgt_lengths = data_lengths
	if USE_GPU:
		src_id = Variable(src_id.cuda())
		tgt_id = Variable(tgt_id.cuda())
		src_mask = Variable(src_mask.cuda())
		tgt_mask = Variable(tgt_mask.cuda())
	else:
		src_id = Variable(src_id)
		tgt_id = Variable(tgt_id)
		src_mask = Variable(src_mask)
		tgt_mask = Variable(tgt_mask)

	pred_ids, _ , _ = greedy_search(
		model,
		src_id,
		src_mask,
		src_lengths,
		tgt_dict,
		config['evaluation']['max_decode_len'],
		USE_GPU
	)

	pred_ids_lst = pred_ids.data.tolist()
	pred_batch_lst = tgt_dict.convert_id_lst_to_symbol_lst(pred_ids_lst)
	cand_lst.extend(pred_batch_lst)
	gold_batch_lst = [tup[1] for tup in data_symbol]
	gold_lst.extend(gold_batch_lst)

	ngram_bleus, bleu, bp, hyp_ref_len, ratio = bleu_calculator.calc_bleu(
		cand_lst,
		[gold_lst]
	)

print('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)'
	% (
		bleu,
		ngram_bleus[0],
		ngram_bleus[1],
		ngram_bleus[2],
		ngram_bleus[3],
		bp,
		ratio,
		hyp_ref_len[0],
		hyp_ref_len[1]
	)
)

# greedy decoding
cand_lst = []
gold_lst = []
for testBatchIdx in tqdm(xrange(len(test_data))):

	data_symbol, data_id, data_mask, data_lengths = test_data[testBatchIdx]
	src_id, tgt_id = data_id
	src_mask, tgt_mask = data_mask
	src_lengths, tgt_lengths = data_lengths
	if USE_GPU:
		src_id = Variable(src_id.cuda())
		tgt_id = Variable(tgt_id.cuda())
		src_mask = Variable(src_mask.cuda())
		tgt_mask = Variable(tgt_mask.cuda())
	else:
		src_id = Variable(src_id)
		tgt_id = Variable(tgt_id)
		src_mask = Variable(src_mask)
		tgt_mask = Variable(tgt_mask)

	pred_ids, _ , _ = greedy_search(
		model,
		src_id,
		src_mask,
		src_lengths,
		tgt_dict,
		config['evaluation']['max_decode_len'],
		USE_GPU
	)

	pred_ids_lst = pred_ids.data.tolist()
	pred_batch_lst = tgt_dict.convert_id_lst_to_symbol_lst(pred_ids_lst)
	cand_lst.extend(pred_batch_lst)
	gold_batch_lst = [tup[1] for tup in data_symbol]
	gold_lst.extend(gold_batch_lst)

	ngram_bleus, bleu, bp, hyp_ref_len, ratio = bleu_calculator.calc_bleu(
		cand_lst,
		[gold_lst]
	)

print('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)'
	% (
		bleu,
		ngram_bleus[0],
		ngram_bleus[1],
		ngram_bleus[2],
		ngram_bleus[3],
		bp,
		ratio,
		hyp_ref_len[0],
		hyp_ref_len[1]
	)
)
