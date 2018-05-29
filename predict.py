from model import *
from torch.autograd import Variable

import torch
import pickle
import sentencepiece as spm
import pretrainedmodels
import pretrainedmodels.utils as utils
import numpy
import json

imodel_name = 'nasnetalarge'
imodel = pretrainedmodels.__dict__[imodel_name](num_classes=1000, pretrained='imagenet')
imodel.eval()
imodel.cuda()
imodel.load_state_dict(torch.load("im_model.pkl"))


sp = spm.SentencePieceProcessor()
sp.Load("titles.spmodel")

with open("title_vocab.json") as json_data:
	title_dictionary = json.load(json_data)
with open("label_vocab.json") as json_data:
	label_dictionary = json.load(json_data)
rlabel_dictionary = {v: k for k, v in label_dictionary.iteritems()}

id_dict = pickle.load(open("id_dict.p", "rb"))
id_name = id_dict[0]
key_id = id_dict[1]

mmodel = Encoder2Decoder(len(title_dictionary), len(label_dictionary), imodel)
mmodel.load_state_dict(torch.load("enc_model.pkl"))
mmodel.cuda()

load_img = utils.LoadImage()
tf_img = utils.TransformImage(imodel)

def predict(image_feature, text_feature, seq=None):
	image_fingerprint = image_feature
	text_fingerprint = mmodel.text_encoder(text_feature)
	fingerprint = mmodel.fuser(image_fingerprint, text_fingerprint)
	states = None
	scores, _ = mmodel.decoder(fingerprint, seq, states)
	sscores = scores.sort()
	pvec = sscores[0][0][-1]
	ivec = sscores[1][0][-1]
	pvec = pvec.tolist()
	avg = 0
	for a in range(len(pvec)-1):
		avg += abs(pvec[a+1]-pvec[a])
	avg = avg / float(len(pvec) - 1)
		
	ivec = ivec.tolist()
	confident_ones = []
	for a in range(len(pvec)-1, 1, -1):
		if pvec[a] - pvec[a-1] > avg:
			confident_ones.append(rlabel_dictionary[ivec[a]])
			break
		else:
			confident_ones.append(rlabel_dictionary[ivec[a]])
		
	return confident_ones

def preprocess(img_dir, tokens, seq):
	input_img = load_img(img_dir)
	input_tensor = tf_img(input_img).cuda()
	input_tensor = input_tensor.unsqueeze(0)
	input = torch.autograd.Variable(input_tensor, requires_grad=False)
	output_features = imodel.features(input)
	image_feature = imodel.logits(output_features).unsqueeze(0)

	text_ids = [0] + [title_dictionary[w] if w in title_dictionary.keys() else 4 for w in tokens] + [1]
	text_feature = Variable(torch.from_numpy(numpy.asarray(text_ids).astype(numpy.int32)).long().cuda().unsqueeze(0))
	print tokens, text_feature
	seq = [label_dictionary[v] for v in seq]
	if seq:
		seq1 = torch.zeros((text_feature.size(0), 1)).long()
		seq2 = torch.from_numpy(numpy.asarray(seq).astype(numpy.int32)).long().unsqueeze(0)
		seq = Variable(torch.cat((seq1, seq2), dim=1).cuda())
	else:
		seq = Variable(torch.zeros((text_feature.size(0), 1)).long().cuda())

	return image_feature, text_feature, seq

if __name__ == "__main__":
	img_dir = "209281.jpg"	
	text = "H&M - Ribbad topp - Vit - Dam".encode("utf-8")
	tokens = sp.EncodeAsPieces(text)
	tokens = [token.decode("utf-8") for token in tokens]
	seq = ['1', 'upper', 'Topwear', 'Top', 'T-Shirt', 'Slim', 'S'] 
		# seq = {'Gender', 'Category', 'Product Type', 'Product Subtype', 'Cut', 'Fit', 'Stretch Factor'} 
		# seq is a list with min 0 and maximum 7 values

	image_feature, text_feature, oseq = preprocess(img_dir, tokens, seq)
	confident_ones = predict(image_feature, text_feature, oseq)
	nconfident_ones = []

	if len(seq) > 1 and len(seq) < 6:
		if len(seq) == 2:
			vtype = "T"
			parent_name = ""
		elif len(seq) == 3:
			vtype = "S"
			parent_name = seq[2]
		elif len(seq) == 4:
			vtype = "C"
			parent_name = seq[3]
		elif len(seq) == 5:
			vtype = "F"
			parent_name = seq[4]
			
		for a in range(len(confident_ones)):
			key = confident_ones[a] + "," + vtype + "," + seq[0] + "," + parent_name
			id = key_id[key]
			value = id + "_" + confident_ones[a]
			nconfident_ones.append(value)

		confident_ones = nconfident_ones	 


	if confident_ones == ['</S>']:
		print "End of Sequence"
	else:
		print confident_ones
