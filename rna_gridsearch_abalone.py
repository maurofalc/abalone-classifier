import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import time

#%% Divisão da base de dados e pré-processamento
time_begin = time.time()

base = pd.read_csv("abalone.csv")
base.info()

attributes = base.iloc[:, 1:9].values
sex = base.iloc[:, 0].values

scaler = MinMaxScaler(feature_range=(-1, 1))
attributes = scaler.fit_transform(attributes)

labelencoder = LabelEncoder()
sex = labelencoder.fit_transform(sex)
sex_dummy = np_utils.to_categorical(sex)
#001 M
#010 I
#100 F
X_train, X_test, y_train, y_test =  train_test_split(attributes, sex_dummy, shuffle=True, random_state=1, test_size=.25)

#%% Construção da topologia da RNA
def createANN(dropout,optimizer, epochs, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation,input_dim=8))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units=neurons, activation=activation))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units=3, activation="softmax"))
    classifier.compile(optimizer=optimizer, loss="categorical_crossentropy",
                          metrics=["categorical_accuracy"])
    classifier.fit(X_train, y_train, batch_size=16,
                      epochs=epochs)
    return classifier

classifier = KerasClassifier(build_fn=createANN)

#%% Definição dos parâmetros de GridSearch e sua execução
parameters={"dropout": [0.1,0.2],
            "epochs": [300, 500],
            "optimizer": ["adam", "sgd"],
            "activation": ["linear", "tanh"],
            "neurons": [8,16,24]}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=4)

grid_search = grid_search.fit(X_train, y_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

time_end = time.time()
print((time_end - time_begin)/60)