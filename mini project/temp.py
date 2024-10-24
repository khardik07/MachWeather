import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('subdataset(1000data).csv')
dataset.drop(["_heatindexm", "_precipm", "datetime_utc"], axis=1, inplace=True)
dataset.dropna(inplace=True)

indexn = dataset[dataset[' _pressurem'] == -9999].index
dataset.drop(indexn, inplace=True)

Y = dataset.iloc[:, -1]
X = dataset.iloc[:, 0:len(dataset.columns)-1]

weather_condition = pd.get_dummies(X[' _conds'])
weather_condition.drop(["Unknown"], axis=1, inplace=True)
X = pd.concat([X, weather_condition], axis=1)
X.drop([" _conds"], axis=1, inplace=True)

def predict():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    
    score = r2_score(y_test, y_prediction)
    print("Temperature prediction Accuracy = ", score * 100)
    plt.hist(y_prediction, facecolor='red', edgecolor='blue', bins=10, range=(5, 35))
    plt.title("Predicted Temperature Histogram")
    plt.show()

def original():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    
    plt.plot(y_test)
    plt.plot(y_prediction)
    plt.title('Original vs Predicted Temperature')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()
