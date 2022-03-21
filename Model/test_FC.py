from FC import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
import os
import sys
import glob
import pandas as pd
from scipy import io
import numpy as np

model = mlp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
	if (dire.startswith('eyeBlink')):
		for mname in (os.listdir(os.path.join(mat_dir + "/" + dire + "/"))):
			if mname.endswith(".mat"):
				input_data = io.loadmat(os.path.join(mat_dir + "/" + dire + "/" + mname))

				input_d.extend(input_data['exp'])

input_d = np.array(input_d)
print(input_d)
input_d = input_d[-10:]


testX = torch.from_numpy(input_d).float()
testX = testX.to(DEVICE)

test = []


test_dict = {}


model = torch.load(sys.argv[3], map_location=DEVICE)

model.eval()
y_pred = model(testX)
pred = y_pred.cpu().detach().numpy()


print(pred)



test_dict = {}
for i in range(18):
	test_dict[list_of_column_names[i]] = tuple(pred[:,i])


df= pd.DataFrame(test_dict)



print(df)

df.to_csv("test.csv", index=None)
