import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc
import seaborn as sns
import glob
from scipy import ndimage
from subprocess import check_output
# import cv2
# import h5py
# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as tls
from scipy import io
import pandas as pd
import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 


pal = sns.color_palette()

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


# labels = label.to_numpy()
list_of_AU = list_of_column_names

print(list_of_column_names[0])


label= np.where(label>0, 1, 0)
labels = label.astype(int)



print(len(labels))



input_d = []
for dire in sorted(os.listdir(mat_dir)):
	if(dire.startswith('eyeBlink')):
		for mname in (os.listdir(os.path.join(mat_dir + "/" + dire + "/"))):
			if mname.endswith(".mat"):
				input_data = io.loadmat(os.path.join(mat_dir + "/" + dire+ "/" + mname))
				

				input_d.extend(input_data['exp'])

print(len(input_d[0]))

input_d = np.array(input_d)

# print(input_d)
input_d_train = input_d[:5590]
labels_train= labels[:5590]

print(input_d_train)
print(labels_train)

X_train, X_test, y_train, y_test = train_test_split(input_d_train, labels_train, test_size=0.2, shuffle =True, random_state=42)




print(len(X_test))

print(len(y_test))




Cs = [1e6, 1e7]
scores = []
# for c in range(len(Cs)):
# 	print(c)
# 	svr = LinearSVC() #epsilon= 0.14, kernel= "linear")
# 	model = MultiOutputRegressor(svr)
# 	model.fit(X_train, y_train)
# 	scores.append(model.score(X_test, y_test))

# print(svr.get_params().keys())

# from sklearn import svm
# from sklearn.model_selection import GridSearchCV, train_test_split

# svrgs_parameters = {
#     'kernel': ['rbf'],
#     'C':     [150000,200000,250000],
#     'gamma': [0.004,0.0045,0.005]
# }

# svr_cv = GridSearchCV(svm.SVR(epsilon= 0.14), svrgs_parameters, cv=8, scoring= 'neg_mean_squared_log_error')
# model = MultiOutputRegressor(svr_cv)
# model.fit(X_train, y_train)
# print("SVR GridSearch score: "+str(svr_cv.best_score_))
# print("SVR GridSearch params: ")
# print(svr_cv.best_params_)
input_d_test = input_d[-10:]
label_test = labels[-10:]

# svr = GridSearchCV(SVR(epsilon= 0.14),svrgs_parameters, cv=8)
# model = MultiOutputRegressor(svr)
# svr.fit(X_train, y_train)
# # with open('model.pkl','wb') as f:
#     # pickle.dump(clf,f)
# print(svr.score(X_test, y_test))
# y_pred = svr.predict(input_d_test)

test = []
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X_train, y_train)
print(multi_target_forest.score(X_test, y_test))
y_pred = multi_target_forest.predict(input_d_test)
# test_df = pd.DataFrame()
print(y_pred)
# test_dict = dict()
test_dict = {}
GT = {}

for i in range(len(list_of_column_names)):
	# list_of_column_names[i] = []

	# list_of_column_names[i].append(y_pred[:,i])
	# test.append(list_of_column_names[i])
	# print(tuple(y_pred[][:,i]))
	test_dict[list_of_column_names[i]] = tuple(y_pred[:,i])
	GT[list_of_column_names[i]] = tuple(label_test[:,i])

	
	# test_df[list_of_AU[i]] = list_of_column_names[i]


print(test_dict)



# for i in range(52):
	# test_dict[list_of_AU[i]] = test[i]


df= pd.DataFrame(test_dict)

gt = pd.DataFrame(GT)

print(df)

df.to_csv("test_n.csv", index=None)
gt.to_csv("gt.csv", index=None)













mse = []

mae = []


for i in range(18):
	MAE= mean_absolute_error(label_test[:,i], y_pred[:,i])
	MSE= mean_squared_error(label_test[:,i], y_pred[:,i])

	print("AU", i, "= MAE:" ,MAE ,"  MSE:",MSE)

	mae.append(MAE)
	mse.append(MSE)


	mae_one = mean_absolute_error(label_test[:,0], y_pred[:,0])
	mae_two = mean_absolute_error(label_test[:,1], y_pred[:,1])
	print(f'MAE for first regressor: {mae_one} - second regressor: {mae_two}')

