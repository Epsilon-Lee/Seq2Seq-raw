import torch
import torch.nn as nn
import torch.nn.functional as F


def nllloss_criterion(dec_dict):
	weight = torch.FloatTensor(dec_dict.size()).fill_(1)
	weight[dec_dict.tgt_specials['<pad>']] = 0
	crit = nn.NLLLoss(
		weight,
		size_average=False
	)
	return crit


def mle_loss(crit, preds, gold):
	"""A combination of F.log and nn.NLLLoss

	Args
	----------
	preds   : N x (dec_L - 1) x vocab_size
	gold    : N x (dec_L - 1)

	return
	----------
	loss    : torch.FloatTensor size: (1)

	"""
	N = preds.size(0)
	log_preds = torch.log(preds)
	# log_preds = preds
	log_preds_flat = log_preds.view(-1, log_preds.size(2))
	gold_flat = gold.view(-1)
	loss = crit(log_preds_flat, gold_flat)
	loss_per_example = loss / N
	return loss_per_example


# def memoryefficient_mle_loss(cirt, pred, gold):
# 	pass