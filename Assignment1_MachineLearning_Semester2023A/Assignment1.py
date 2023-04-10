### Data Fields
# %% Import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
# %% Load dataset
patient_frame = pd.read_csv('Paitients_Files_Train.csv')

# %% Shape
print(patient_frame.shape)
# %% Head
patient_frame.head()

# %% Info
patient_frame.info()
# %% Descriptions
patient_frame.describe()

# %% Class distributions of Sepssis
patient_frame.groupby('Sepssis').size()

#%% Maximum value of each attribute
pd.DataFrame.max(patient_frame)
# %% Minimum value of each attribute
pd.DataFrame.min(patient_frame)
#%% Mean value of attributes
pd.DataFrame.mean(patient_frame)
#%% Median value of attributes
pd.DataFrame.median(patient_frame)

# %% Histograms
plt.figure()
patient_frame.hist(figsize=(8,20))
plt.show()

# %% box and whisker plots
patient_frame.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(8,20))
plt.show()

# %% Scatter plot matrix
scatter_matrix(patient_frame, figsize=(8,40))
# %% Sk and M11
from sklearn.model_selection import train_test_split
uniRmX = patient_frame['M11'] # no. of rooms
Y = patient_frame['SK']

trainX, testX, trainY, testY = train_test_split(np.array(uniRmX).reshape(-1, 1), np.array(Y).reshape(-1, 1), test_size=0.2)

#%% Try 2 models
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Model 1
poly1 = PolynomialFeatures(1)
trainX_poly1 = poly1.fit_transform(trainX)
model_poly1 = linear_model.LinearRegression()
model_poly1.fit(trainX_poly1, trainY)

# Model 2
poly4 = PolynomialFeatures(4)
trainX_poly4 = poly4.fit_transform(trainX)
model_poly4 = linear_model.LinearRegression()
model_poly4.fit(trainX_poly4, trainY)

#%% Plot data and hypotheses
plt.scatter(testX, testY, color='k')
x_plot = np.arange(4,9,0.2).reshape(-1,1)

x_predict = poly1.fit_transform(x_plot)
pred_poly1 = model_poly1.predict(x_predict)
plt.plot(x_plot, pred_poly1, color='g', linewidth=3)

x_predict = poly4.fit_transform(x_plot)
pred_poly4 = model_poly4.predict(x_predict)
plt.plot(x_plot, pred_poly4, color='b', linewidth=3)
# %%
