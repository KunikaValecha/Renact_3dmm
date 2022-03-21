from FC import mlp
from torch.optim import SGD, ASGD, Adam, Adagrad, SparseAdam, Adamax, AdamW
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
import sys
import os
import glob
import pandas as pd
from scipy import io
import numpy as np
from matplotlib import pyplot as plt





def next_batch(inputs, targets, batchSize):
	for i in range(0, inputs.shape[0], batchSize):
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])





BATCH_SIZE =64
print(BATCH_SIZE)
EPOCHS = 80
LR = 1e-2


class customLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
    	for p in pred:
    		# for k in p:
	    	if all(k < 1 and k>0 for k in p):
	    		return self.mse(pred, actual)
	    	elif any(k>1 for k in p):
	    		q =torch.where(pred>1, pred, torch.zeros_like(pred))
	    		# print(q)
	    		return(self.mse(pred, actual) + (torch.mul(self.mse(q, torch.ones_like(q)),10)))
	    	elif any(k<0 for k in p):
	    		t =torch.where(pred<0, pred, torch.zeros_like(pred))
	    		# print(t)
	    		return(self.mse(pred, actual) + (torch.mul(self.mse(t, torch.zeros_like(t)),10)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

print("[INFO] preparing data...")

csv_dir = sys.argv[1]
mat_dir = sys.argv[2]


label =[]

joined_files = os.path.join(csv_dir, "eye_blink*.csv")
joined_list = glob.glob(joined_files)

label = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
first_column = label.columns[0]
label = label.drop([first_column], axis=1)

list_of_column_names = []

 
for row in label:

    list_of_column_names.append(row)


labels = label.to_numpy()
list_of_AU = list_of_column_names



input_d = []
for dire in sorted(os.listdir(mat_dir)):
	if(dire.startswith('eyeBlink')):
		for mname in (os.listdir(os.path.join(mat_dir + "/" + dire + "/"))):
			if mname.endswith(".mat"):
				input_data = io.loadmat(os.path.join(mat_dir + "/" + dire+ "/" + mname))
				

				input_d.extend(input_data['exp'])

print(len(input_d[0]))

input_d = np.array(input_d)


input_d_train = input_d[:5590]
labels_train= labels[:5590]

(trainX, testX, trainY, testY) = train_test_split(input_d, labels, test_size=0.15, shuffle =True, random_state=95)

trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

mlp = mlp.get_training_model().to(DEVICE)

opt = Adam(mlp.parameters(), lr=LR)
lossFunc = customLoss()




loss_values = []
for epoch in range(0, EPOCHS):
	print("[INFO] epoch: {}...".format(epoch + 1))
	trainLoss = 0
	trainAcc = 0
	samples = 0
	mlp.train()
	
	
	for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
		epoch_loss = []

		(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
		predictions = mlp(batchX)
		# print(predictions)
		loss = lossFunc(predictions, batchY)

		# for i in
		
		# if any(x < 0 for x in predictions):
		
		# 	loss = lossFunc(predictions, batchY) + (lossFunc(predictions, 0) *10)


		# elif all(x>1 for x in predictions):
		# 	loss = lossFunc(predictions, batchY) + (lossFunc(predictions, 0) *10)

		# else:
		# 	loss = lossFunc(predictions, batchY)

		
		opt.zero_grad()
		loss.backward()
		opt.step()
		trainLoss += loss.item() * batchY.size(0)
		
		
		
		samples += batchY.size(0)


	trainTemplate = "epoch: {} train loss: {:.3f}" 
	print(trainTemplate.format(epoch + 1, (trainLoss / samples)))
	epoch_loss = trainLoss / samples
	loss_values.append(epoch_loss)

	
	# initialize tracker variables for testing, then set our model to
	# evaluation mode
	testLoss = 0
	testAcc = 0
	samples = 0
	mlp.eval()

	# initialize a no-gradient context
	with torch.no_grad():
		# loop over the current batch of test data
		for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
			# flash the data to the current device
			(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

			# run data through our model and calculate loss
			predictions = mlp(batchX)
			loss = lossFunc(predictions, batchY)
			# if predictions<1 or predictions>0:
		
			# 	loss = lossFunc(predictions, batchY)

			# else:
			# 	loss = lossFunc(predictions, batchY)*10
			# update test loss, accuracy, and the number of
			# samples visited
			testLoss += loss.item() * batchY.size(0)
			samples += batchY.size(0)

		# display model progress on the current test batch
		testTemplate = "epoch: {} test loss: {:.3f}"
		print(testTemplate.format(epoch + 1, (testLoss / samples),
			(testAcc / samples)))
		print("")

torch.save(mlp, sys.argv[3])
            
		
