import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

gsc = GridSearchCV(
    estimator=SVR(kernel='poly'),
    param_grid={
        'C': [1, 10, 100, 1000, 1.5, 2, 5],
        'epsilon': [0.1, 0.2, 0.3, 0.5, 0.05],
        'gamma': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
)

grid_result = gsc.fit(XTrain, yTrain)
best_params = grid_result.best_params_

NAPsvm = SVR(
    kernel='poly',
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    gamma=best_params["gamma"],
    coef0=0.1,
    shrinking=True,
    tol=0.001,
    cache_size=200,
    verbose=False,
    max_iter=-1
)

NAPsvm.fit(XTrain, yTrain)

yPredcit = NAPsvm.predict(XTest)

mse = mean_squared_error(yTest, yPredcit, squared=True)
rmse = mean_squared_error(yTest, yPredcit, squared=False)
mae = mean_absolute_error(yTest, yPredcit)

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root Mean Square Error (RMSE) on test set: {:.4f}".format(rmse))
print("The mean absolute error on test set: {:.4f}".format(mae))

yPredcitnew = NAPsvm.predict(minmaxScaler.transform([[20, 1050, 30]]))
print(yPredcitnew)
