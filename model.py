import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import pretrainedmodels
import pretrainedmodels.utils as utils
import sys

#from DataIterator import *

# Image Encoder
class ImageEncoder(nn.Module):
	def __init__(self, image_model):
		super(ImageEncoder, self).__init__()

		self.model = image_model

	def forward(self, input_tensor):

		# transformations
		#tf_img = utils.TransformImage(self.model)

		#input_tensor = tf_img(image)
		#input_tensor = input_tensor.unsqueeze(0).cuda()
		input = torch.autograd.Variable(input_tensor, requires_grad=False)
		output_features = self.model.features(input)
		image_embedding = self.model.logits(output_features).unsqueeze(1)

		return image_embedding

# Text Encoder
class TextEncoder(nn.Module):
	def __init__(self, dict_size):
		super(TextEncoder, self).__init__()

		# word embedding
		self.embedding = nn.Embedding(dict_size, 1000)

		# GRU encoder
		self.GRU = nn.GRU(1000, 1000, 1, batch_first=True)

	def forward(self, text, states=None):

		# Word embedding
		word_embeddings = self.embedding(text)

		_, text_fingerprint = self.GRU(word_embeddings, states)

		text_fingerprint = text_fingerprint.transpose(0, 1)

		return text_fingerprint

# Fingerprint Fuser
class Fuser(nn.Module):
	def __init__(self):
		super(Fuser, self).__init__()

		# Linear Transformer
		self.fuser = nn.Linear(2000, 1000)

	def forward(self, image_fingerprint, text_fingerprint):

		# Linear transformation for fusing image and text fingerprints
		fingerprint = torch.cat((image_fingerprint, text_fingerprint), dim=2)
		fingerprint = self.fuser(fingerprint)

		return fingerprint

# Caption Decoder
class Decoder(nn.Module):
	def __init__(self, dict_size):
        	super( Decoder, self ).__init__()

        	# word embedding
        	self.embed = nn.Embedding(dict_size, 1000)

        	# GRU decoder
        	self.GRU = nn.GRU(1000, 1000, 1, batch_first=True)

		# Final classifier
		self.mlp = nn.Linear(1000, dict_size)

		self.init_weights()

	def init_weights(self):
        	'''
        	Initialize final classifier weights
        	'''
        	init.kaiming_normal(self.mlp.weight, mode='fan_in')
		self.mlp.bias.data.fill_(0)

    	def forward(self, fingerprint, captions, states=None):

        	# Word Embedding
        	word_embeddings = self.embed(captions)

        	x = torch.cat((fingerprint, word_embeddings), 1)

        	output, states = self.GRU(x, states)

		score = self.mlp(output)

        	# Return states for Caption Sampling purpose
        	return score, states

# Whole Architecture
class Encoder2Decoder( nn.Module ):
    def __init__(self, text_dict_size, label_dict_size, image_model):
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder and Adaptive Attention Decoder
        self.image_encoder = ImageEncoder(image_model)
	self.text_encoder = TextEncoder(text_dict_size)
	self.fuser = Fuser()
	self.decoder = Decoder(label_dict_size)

    def forward(self, images, text, captions, lengths):

        image_fingerprints = self.image_encoder(images)
	text_fingerprints = self.text_encoder(text)
	fingerprint = self.fuser(image_fingerprints, text_fingerprints)

        # Language Modeling on word prediction
        scores, _ = self.decoder(fingerprint, captions)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores

    # Caption generator
    def sampler(self, images, text, max_len=8):
        """
        Samples captions for given image and text features (Greedy search).
        """

	image_fingerprint = self.image_encoder(images)
	text_fingerprint = self.text_encoder(Variable(text))
	fingerprint = self.fuser(image_fingerprint, text_fingerprint)

	# Build the starting token Variable <start> (index 1): B x 1
        captions = Variable(torch.zeros((fingerprint.size(0), 1)).long().cuda())

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
	conf = []

        # Initial hidden states
        states = None

        for i in range(max_len):

            scores, states = self.decoder(fingerprint, captions, states)
	    vec = scores.sort()[0][0][-1]
	    if (vec[-1].data.tolist()[0] - vec[-2].data.tolist()[0]) > vec.std().data.tolist()[0]:
		conf.append("confident")
	    else:
		conf.append("not_confident")
	    predicted = scores.max(2)[1][:, -1]
            captions = torch.cat((captions, predicted), dim=1)

            # Save sampled word, attention map and sentinel at each timestep
            #sampled_ids.append(captions)

        # caption: B x max_len

        return captions, conf
