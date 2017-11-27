import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from torch.autograd import Variable
import torch.optim as optim

from seq2seq.seq2seq_models import Seq2Seq, BahdahnauAttentionSeq2Seq, GlobalAttentionSeq2Seq
from seq2seq.Dict import Dict
from seq2seq.bleucal import BleuCalculator
from seq2seq.modules.loss import nllloss_criterion, mle_loss
from seq2seq.search import greedy_search

import json
import time
import os
import math
from tqdm import tqdm

# Global variables
USE_GPU = False
PARAM_INIT = 0.1

# 1. Get config object
config_file_path = './config_de_en_vanilla_iwslt.json'
config = json.load(open(config_file_path, 'r'))

# 2. Set gpu device
if cuda.is_available():
	USE_GPU = True
	cuda.set_device(config['management']['gpuid'])

# 3. Load train, dev, test data and print dataset info
train_data = torch.load(config['data']['train_path'])
dev_data   = torch.load(config['data']['dev_path'])
test_data  = torch.load(config['data']['test_path'])

train_data.set_batch_size(config['training']['train_batch_size'])
dev_data.set_batch_size(config['training']['dev_batch_size'])
test_data.set_batch_size(config['training']['test_batch_size'])

print("========== Dataset info ==========")
print("1. Training data")
print("batch size  : %d" % train_data.batch_size)
print("batch num   : %d" % train_data.batch_num)
print("sentence num: %d" % len(train_data.data_symbol))
print("2. Training data")
print("batch size  : %d" % dev_data.batch_size)
print("batch num   : %d" % dev_data.batch_num)
print("sentence num: %d" % len(dev_data.data_symbol))
print("3. Training data")
print("batch size  : %d" % test_data.batch_size)
print("batch num   : %d" % test_data.batch_num)
print("sentence num: %d" % len(test_data.data_symbol))
print

# 4. Load src, tgt dictionary
src_dict   = Dict(None, config['data']['src_dict_path'])
tgt_dict   = Dict(None, config['data']['tgt_dict_path'])

# 5. Create model and initialize the parameters
# Bahdahnau attention model
# model = BahdahnauAttentionSeq2Seq(
# 	src_dict,
# 	src_dict.src_specials['<pad>'],
# 	config['model']['encoder']['emb_size'],
# 	config['model']['encoder']['hid_size'],
# 	config['model']['encoder']['bidirectional'],
# 	config['model']['encoder']['rnn_cell_type'],
# 	config['model']['encoder']['is_packed'],
# 	config['model']['encoder']['batch_first'],
# 	config['model']['encoder']['num_layers'],
# 	config['model']['encoder']['dropout'],
# 	tgt_dict,
# 	tgt_dict.tgt_specials['<pad>'],
# 	config['model']['decoder']['emb_size'],
# 	config['model']['decoder']['hid_size'],
# 	config['model']['decoder']['rnn_cell_type'],
# 	config['model']['decoder']['num_layers'],
# 	config['model']['decoder']['dropout'],
# 	config['model']['generator']['dim_lst'],
# 	config['model']['generator']['num_layers']
# )

# Global attention model
model = GlobalAttentionSeq2Seq(
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
	config['model']['decoder']['global_attention_type'],
	config['model']['generator']['dim_lst'],
	config['model']['generator']['num_layers']
)

# # Naive Seq2Seq model
# model = Seq2Seq(
# 	src_dict,
# 	src_dict.src_specials['<pad>'],
# 	config['model']['encoder']['emb_size'],
# 	config['model']['encoder']['hid_size'],
# 	config['model']['encoder']['bidirectional'],
# 	config['model']['encoder']['rnn_cell_type'],
# 	config['model']['encoder']['is_packed'],
# 	config['model']['encoder']['batch_first'],
# 	config['model']['encoder']['num_layers'],
# 	config['model']['encoder']['dropout'],
# 	tgt_dict,
# 	tgt_dict.tgt_specials['<pad>'],
# 	config['model']['decoder']['emb_size'],
# 	config['model']['decoder']['hid_size'],
# 	config['model']['decoder']['rnn_cell_type'],
# 	config['model']['decoder']['num_layers'],
# 	config['model']['decoder']['dropout'],
# 	config['model']['generator']['dim_lst'],
# 	config['model']['generator']['num_layers']
# )

for param in model.parameters():
	param.data.uniform_(-PARAM_INIT, PARAM_INIT)

print model

# 6. Create optimizer
if config['training']['optimizer'] == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=config['training']['lr'])
elif config['training']['optimizer'] == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

# 7. Create BLEU calculator
bleu_calulator = BleuCalculator(None, None)

# 8. Create loss criterion
criterion = nllloss_criterion(tgt_dict)

# 9. Sent to gpu
if USE_GPU:
	model.cuda()
	criterion.cuda()

# 10. Open log file and start train-valid iteration
if os.path.exists(config['management']['logfile_path']):
	f_log = open(config['management']['logfile_path'], 'a') # append to old file
else:
	f_log = open(config['management']['logfile_path'], 'w') # create new file
f_log.write("\n")
f_log.write('------------------------Start Experiment------------------------\n')
f_log.write(time.asctime(time.localtime(time.time())) + "\n")
f_log.write('----------------------------------------------------------------\n')
f_log.write('================= config =================\n')
f_log.write('| Model name: %s |\n' % model.name)
if model.name == 'GlobalAttentionSeq2Seq':
	f_log.write('| Attention type: %s |\n' % config['model']['decoder']['global_attention_type'])
f_log.write('---- Encoder ----\n')
f_log.write('RNN_cell_type    : %s\n' % config['model']['encoder']['rnn_cell_type'])
f_log.write('Embedding size   : %d\n' % config['model']['encoder']['emb_size'])
f_log.write('Hidden size      : %d\n' % config['model']['encoder']['hid_size'])
f_log.write('Number of layers : %d\n' % config['model']['encoder']['num_layers'])
f_log.write('Bidirectional    : %d\n' % config['model']['encoder']['bidirectional'])
f_log.write('Dropout rate     : %f\n' % config['model']['encoder']['dropout'])
f_log.write('---- Decoder ----\n')
f_log.write('RNN_cell_type    : %s\n' % config['model']['decoder']['rnn_cell_type'])
f_log.write('Embedding size   : %d\n' % config['model']['encoder']['emb_size'])
f_log.write('Hidden size      : %d\n' % config['model']['encoder']['hid_size'])
f_log.write('Number of layers : %d\n' % config['model']['encoder']['num_layers'])
f_log.write('Dropout rate     : %f\n' % config['model']['encoder']['dropout'])
f_log.write('---- Optimizer ----\n')
f_log.write('Optimizer        : %s\n' % config['training']['optimizer'])
f_log.write('Learning rate    : %f\n' % config['training']['lr'])
f_log.write('Gradient clip    : %f\n' % config['training']['grad_threshold'])
f_log.write('================= config =================\n')
f_log.write('\n')

GRAD_C = config['training']['grad_threshold']
acc_count_total = 0
word_count_total = 0
loss_total = 0.
start_time = time.time()
for epochIdx in xrange(config['training']['max_epoch']):
	
	for batchIdx in xrange(len(train_data)):
		
		data_symbol, data_id, data_mask, data_lengths = train_data[batchIdx]
		src_id, tgt_id = data_id
		src_mask, tgt_mask = data_mask
		src_lengths, tgt_lengths = data_lengths
		if USE_GPU:
			src_id = Variable(src_id).cuda()
			tgt_id = Variable(tgt_id).cuda()
			src_mask = Variable(src_mask).cuda()
			tgt_mask = Variable(tgt_mask).cuda()
		else:
			src_id = Variable(src_id)
			tgt_id = Variable(tgt_id)
			src_mask = Variable(src_mask)
			tgt_mask = Variable(tgt_mask)
		
		model.zero_grad()
		
		# # 1. Naive Seq2Seq
		# preds = model(
		# 	src_id,
		# 	src_mask,
		# 	src_lengths,
		# 	tgt_id,
		# 	tgt_mask,
		# 	tgt_lengths
		# ) # N x (dec_L - 1) x V , N x enc_L x dec_L

		# 2. Attentive Seq2Seq
		preds, atts = model(
			src_id,
			src_mask,
			src_lengths,
			tgt_id,
			tgt_mask,
			tgt_lengths
		) # N x (dec_L - 1) x V , N x enc_L x dec_L

		loss = mle_loss(criterion, preds, tgt_id[:, 1:].contiguous())

		loss.backward()

		# gradient clipping
		grad_norms = []
		params = model.parameters()
		for param in params:
			grad_norm = param.grad.norm().data[0]
			grad_norms.append(grad_norm)
			if grad_norm > GRAD_C:
				param.grad.data = param.grad.data * GRAD_C / grad_norm
		grad_norm_avg = sum(grad_norms) / len(grad_norms)

		optimizer.step()

		# model predict: golden trajectory
		_ , pred_ids = torch.max(preds, dim=2) # N x (dec_L - 1)
		acc_count_batch = pred_ids.data.eq(tgt_id[:, 1:].data).masked_select(
			tgt_id[:, 1:].data.ne(tgt_dict.tgt_specials['<pad>'])
		).sum()

		# except the last </s> symbol
		word_count_batch = sum(tgt_lengths) - len(tgt_lengths)

		# print(type(acc_count_batch))
		acc_batch = acc_count_batch * 1. / word_count_batch

		acc_count_total += acc_count_batch
		word_count_total += word_count_batch

		acc_avg = acc_count_total * 1. / word_count_total

		loss_per_example = loss.data[0]
		loss_per_word = loss_per_example * src_id.size(0) / word_count_batch

		loss_total += loss_per_example * src_id.size(0)
		loss_per_word_avg = loss_total / word_count_total

		# logging
		if (batchIdx + 1) % config['management']['logging_interval'] == 0:
			print("Epoch %d Batch %d loss %f ppl %f acc: %s acc_avg: %s grad_norm: %f time elapsed: %f"
				% (
					epochIdx + 1,
					batchIdx + 1,
					loss_per_word,
					math.exp(loss_per_word_avg),
					format(acc_batch, "2.2%"),
					format(acc_avg, "2.2%"),
					grad_norm_avg,
					time.time() - start_time
				)
			)
			f_log.write("Epoch %d Batch %d loss %f ppl %f acc: %s acc_avg: %s grad_norm: %f time elapsed: %f\n"
				% (
					epochIdx + 1,
					batchIdx + 1,
					loss_per_word,
					math.exp(loss_per_word_avg),
					format(acc_batch, "2.2%"),
					format(acc_avg, "2.2%"),
					grad_norm_avg,
					time.time() - start_time
				)
			)

		# visualization
		if (batchIdx + 1) % config['management']['print_samples'] == 0:
			print('')
			print('--------------------------------------------------------------------------')
			print('Visulize some examples predicted by the model with gold feed-in trajectory')
			print('--------------------------------------------------------------------------')
			f_log.write('--------------------------------------------------------------------------\n')
			f_log.write('Visulize some examples predicted by the model with gold feed-in trajectory\n')
			f_log.write('--------------------------------------------------------------------------\n')
			pred_ids_lst = pred_ids.data.tolist()
			src_symbols_lst = [gold_tup[0] for gold_tup in data_symbol]
			gold_symbols_lst = [gold_tup[1] for gold_tup in data_symbol]
			for i in range(config['management']['print_number']):
				pred_ids = pred_ids_lst[i][:tgt_lengths[i]]
				pred_sent = " ".join(tgt_dict.convertIdxSeq2SymbolSeq(pred_ids))
				src_sent = src_symbols_lst[i]
				gold_sent = gold_symbols_lst[i]
				print('src_gold:')
				print(src_sent)
				print('tgt_pred:')
				print(pred_sent)
				print('tgt_gold:')
				print(gold_sent)
				print
				f_log.write('src_gold:\n')
				f_log.write(src_sent + '\n')
				f_log.write('tgt_pred:\n')
				f_log.write(pred_sent + '\n')
				f_log.write('tgt_gold:\n')
				f_log.write(gold_sent + '\n')
				f_log.write('\n')
			print('--------------------------------------------------------------------------')
			f_log.write('--------------------------------------------------------------------------\n')

		# evaluation
		# 1. Greedy decoding
		if (batchIdx + 1) % config['management']['eval_interval'] == 0:
			print('----------Evaluation on dev set----------')
			f_log.write('----------Evaluation on dev set----------\n')
			cand_lst = []
			gold_lst = []
			for devBatchIdx in tqdm(xrange(len(dev_data))):
				data_symbol, data_id, data_mask, data_lengths = dev_data[devBatchIdx]
				src_id, tgt_id = data_id
				src_mask, tgt_mask = data_mask
				src_lengths, tgt_lengths = data_lengths
				if USE_GPU:
					src_id = Variable(src_id).cuda()
					tgt_id = Variable(tgt_id).cuda()
					src_mask = Variable(src_mask).cuda()
					tgt_mask = Variable(tgt_mask).cuda()
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
				) # N x max_decode_len
				pred_ids_lst = pred_ids.data.tolist()
				pred_batch_lst = tgt_dict.convert_id_lst_to_symbol_lst(pred_ids_lst)
				cand_lst.extend(pred_batch_lst)
				# single ref. 
				gold_batch_lst = [tup[1] for tup in data_symbol]
				gold_lst.extend(gold_batch_lst)
			
			ngram_bleus, bleu, bp, hyp_ref_len, ratio = bleu_calulator.calc_bleu(
				cand_lst,
				[gold_lst]
			)
			print('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)'
				% (
					bleu * 100,
					ngram_bleus[0] * 100,
					ngram_bleus[1] * 100,
					ngram_bleus[2] * 100,
					ngram_bleus[3] * 100,
					bp,
					ratio,
					hyp_ref_len[0],
					hyp_ref_len[1]
				)
			)
			f_log.write('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)\n'
				% (
					bleu * 100,
					ngram_bleus[0] * 100,
					ngram_bleus[1] * 100,
					ngram_bleus[2] * 100,
					ngram_bleus[3] * 100,
					bp,
					ratio,
					hyp_ref_len[0],
					hyp_ref_len[1]
				)
			)

			print('----------Evaluation on test set----------')
			f_log.write('----------Evaluation on test set----------\n')
			cand_lst = []
			gold_lst = []
			for testBatchIdx in tqdm(xrange(len(test_data))):
				data_symbol, data_id, data_mask, data_lengths = test_data[testBatchIdx]
				src_id, tgt_id = data_id
				src_mask, tgt_mask = data_mask
				src_lengths, tgt_lengths = data_lengths
				if USE_GPU:
					src_id = Variable(src_id).cuda()
					tgt_id = Variable(tgt_id).cuda()
					src_mask = Variable(src_mask).cuda()
					tgt_mask = Variable(tgt_mask).cuda()
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
				) # N x max_decode_len
				pred_ids_lst = pred_ids.data.tolist()
				pred_batch_lst = tgt_dict.convert_id_lst_to_symbol_lst(pred_ids_lst)
				cand_lst.extend(pred_batch_lst)
				# single ref. 
				gold_batch_lst = [tup[1] for tup in data_symbol]
				gold_lst.extend(gold_batch_lst)
			
			ngram_bleus, bleu, bp, hyp_ref_len, ratio = bleu_calulator.calc_bleu(
				cand_lst,
				[gold_lst]
			)
			print('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)'
				% (
					bleu * 100,
					ngram_bleus[0] * 100,
					ngram_bleus[1] * 100,
					ngram_bleus[2] * 100,
					ngram_bleus[3] * 100,
					bp,
					ratio,
					hyp_ref_len[0],
					hyp_ref_len[1]
				)
			)
			f_log.write('BLEU: %2.2f (%2.2f, %2.2f, %2.2f, %2.2f) BP: %.5f ratio: %.5f (%d/%d)\n'
				% (
					bleu * 100,
					ngram_bleus[0] * 100,
					ngram_bleus[1] * 100,
					ngram_bleus[2] * 100,
					ngram_bleus[3] * 100,
					bp,
					ratio,
					hyp_ref_len[0],
					hyp_ref_len[1]
				)
			)
			
		## 2. Beam search decoding

	# save checkpoint
	if USE_GPU:
		model_state_dict = {}
		for name, param in model.state_dict().iteritems():
			model_state_dict[name] = param.cpu()
	else:
		model_state_dict = model.state_dict()
	torch.save(
		model_state_dict,
		'../Models/Seq2Seq-raw_Models/modular_global_bid_%s.pt' % str(epochIdx)
	)
