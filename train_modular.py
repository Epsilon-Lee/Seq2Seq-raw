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

import json
import time


# Global variables
USE_GPU = False


# 1. Get config object
config_file_path = './config_de_en_vanilla_iwslt.json'
config = json.load(open(config_file_path, 'r'))

# 2. Set gpu device
if cuda.is_available():
	USE_GPU = True
	cuda.set_device(config['management']['gpuid'])

# 3. Load train, dev, test data
train_data = torch.load(config['data']['train_path'])
dev_data   = torch.load(config['data']['dev_path'])
test_data  = torch.load(config['data']['test_path'])

# 4. Load src, tgt dictionary
src_dict   = Dict(None, config['data']['src_dict_path'])
tgt_dict   = Dict(None, config['data']['tgt_dict_path'])

# 5. Create model
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

# 6. Create optimizer
if config['training']['optimizer'] == 'sgd':
	optimizer = optim.SGD(model.parameters(), lr=config['training']['lr'])
elif config['training']['optimizer'] == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

# 7. Create BLEU calculator
bleu_calulator = BleuCalculator(None, None)

# 8. Create loss criterion
criterion = nllloss_critterion(tgt_dict)

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
f_log.write('================= config =================')
f_log.write('---- Encoder ----')
f_log.write('RNN_cell_type    : %s' % config['model']['encoder']['rnn_cell_type'])
f_log.write('Embedding size   : %d' % config['model']['encoder']['emb_size'])
f_log.write('Hidden size      : %d' % config['model']['encoder']['hid_size'])
f_log.write('Number of layers : %d' % config['model']['encoder']['num_layers'])
f_log.write('Bidirectional    : %d' % config['model']['encoder']['bidirectional'])
f_log.write('Dropout rate     : %f' % config['model']['encoder']['dropout'])
f_log.write('---- Decoder ----')
f_log.write('RNN_cell_type    : %s' % config['model']['decoder']['rnn_cell_type'])
f_log.write('Embedding size   : %d' % config['model']['encoder']['emb_size'])
f_log.write('Hidden size      : %d' % config['model']['encoder']['hid_size'])
f_log.write('Number of layers : %d' % config['model']['encoder']['num_layers'])
f_log.write('Dropout rate     : %f' % config['model']['encoder']['dropout'])
f_log.write('---- Optimizer ----')
f_log.write('Optimizer        : %s' % config['training']['optimizer'])
f_log.write('Learning rate    : %f' % config['training']['lr'])
f_log.write('Gradient clip    : %f' % config['training']['grad_threshold'])
f_log.write('================= config =================')
f_log.write('\n')

GRAD_C = config['training']['grad_threshold']
acc_count_total = 0
word_count_total = 0
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
		preds, atts = model(
			src_id,
			src_mask,
			src_lengths,
			tgt_id,
			tgt_mask,
			tgt_lengths
		) # N x (dec_L - 1) x V , N x enc_L x dec_L

		loss = mle_loss(criterion, preds, tgt_id[:, 1:])

		loss.backward()

		# gradient clipping
		grad_norms = []
		params = model.parameters()
		for param in params:
			grad_norm = param.grad.norm().data[0]
			grad_norms.append(grad_norm)
			if grad_norm > GRAD_C:
				param.grad.data = para.grad.data * GRAD_C / grad_norm
		grad_norm_avg = sum(grad_norms) / len(grad_norms)

		optimizer.step()

		# model predict: golden trajectory
		_ , pred_ids = torch.max(preds, dim=2) # N x (dec_L - 1)
		acc_count_batch = pred_ids.eq(tgt_id[:, 1:]).masked_select(
			tgt_id[:, 1:].ne(tgt_dict.tgt_specials['<pad>'])
		).sum()

		# except the last </s> symbol
		word_count_batch = sum(tgt_lengths) - len(tgt_lengths)

		acc_batch = acc_count_batch * 1. / word_count_batch

		acc_count_total += acc_count_batch
		word_count_total += word_count_batch

		acc_avg = acc_count_total * 1. / word_count_total

		loss_per_example = loss.data[0]
		loss_per_word = loss_per_example * src_id.size(0) / word_count_batch

		loss_total += loss_per_example * src_id.size(0)
		loss_per_word_avg = loss_total / word_count_total

		# logging
		if (batchIdx + 1) % config['management']['logging_intervl'] == 0:
			print("Epoch %d Batch %d loss %f ppl %f acc: %s acc_avg: %s grad_norm: %f time elapsed: %f"
				% (
					epochIdx,
					batchIdx,
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
					epochIdx,
					batchIdx,
					loss_per_word,
					math.exp(loss_per_word_avg),
					format(acc_batch, "2.2%"),
					format(acc_avg, "2.2%"),
					grad_norm_avg,
					time.time() - start_time
				)
			)

		# visualize
		# if (batchIdx + 1) % config['management']['print_samples'] == 0:
		# 	pred_ids_lst = pred_ids.tolist()
		# 	for i in range(config['management']['print_number']):

