import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier


dataset = pd.read_csv('FinalDSPER.csv')

# set for the values to train and test percentages
TrainPercentage = 0.7
TestPercentage = 0.3

nor = dataset['HostName'].count()

# define  X,Y
X = dataset.iloc[:nor, [1, 2, 3]].values
Y = dataset.iloc[:nor, [5]].values

# standardization (Normalization)
minmaxScaler = MinMaxScaler()
XStandard = minmaxScaler.fit_transform(X)

XTrain, XTest, yTrain, yTest = train_test_split(XStandard, Y, test_size=TestPercentage, random_state=42)

# change the shape of y to (n_samples, )
yTrain = yTrain.ravel()

# train the model (SVR)
from sklearn.svm import SVR



NAPxgb = XGBClassifier()

NAPxgb.fit(XTrain, yTrain)

yPredcit = NAPxgb.predict(XTest)
print(accuracy_score(yTest,yPredcit))

mse = mean_squared_error(yTest, yPredcit, squared=True)
rmse = mean_squared_error(yTest, yPredcit, squared=False)
mae = mean_absolute_error(yTest, yPredcit)

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root Mean Square Error (RMSE) on test set: {:.4f}".format(rmse))
print("The mean absolute error on test set: {:.4f}".format(mae))

yPredcitnew = NAPxgb.predict(minmaxScaler.transform([[20, 1050, 30]]))
print(yPredcitnew)
