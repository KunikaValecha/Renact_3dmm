from collections import OrderedDict
import torch.nn as nn
def get_training_model(inFeatures=64, hiddenDim=128, nbClasses=18):
	# construct a shallow, sequential neural network
	mlpModel = nn.Sequential(OrderedDict([
		("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
		("activation_1", nn.ReLU()),
		# ("hidden_layer_2", nn.Linear(128,256)),
		# ("activation_2", nn.ReLU()),
		# ("hidden_layer_3", nn.Linear(256, 512)),
		# ("activation_3", nn.ReLU()),
		("output_layer", nn.Linear(128, nbClasses))
	]))
	# return the sequential model
	return mlpModel
