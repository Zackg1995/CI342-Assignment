import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from matplotlib import style

houseData = pd.read_csv("Livingroomdataset.csv", index_col=0)
houseData = houseData[["Rooms","P2019", "P2020", "P2021"]]

pricePrediction = "P2021"

x_axis = np.array(houseData.drop([pricePrediction], 1))
y_axis = np.array(houseData[pricePrediction])
x_train,  x_test,y_train, y_test = sklearn.model_selection.train_test_split(x_axis,y_axis, test_size=0.1)

bestPrediction = 0

for _ in range(50) :
    x_train,  x_test,y_train, y_test = sklearn.model_selection.train_test_split(x_axis,y_axis, test_size=0.1)

houseModel = linear_model.LinearRegression()

houseModel.fit(x_train, y_train)

predictionAcc = houseModel.score(x_test, y_test)

print(predictionAcc)

if predictionAcc > bestPrediction:
    best = predictionAcc
with open("pricingModel.pickle", "wb") as f:
    pickle.dump(houseModel,f)

picklein = open("pricingModel.pickle", "rb")
houseModel = pickle.load(picklein)

housePricePredictions = houseModel.predict(x_test)

for x in range (len(housePricePredictions)):
    print(housePricePredictions[x], x_test[x],y_test[x])


style.use("ggplot")
plt.scatter(houseData["P2019"], houseData["P2021"], s = 100, c = "green", edgecolors="black", linewidths="1", alpha=0.75)
plt.xlabel("P2019")
plt.ylabel("2021 Price Prediction")
plt.show()
