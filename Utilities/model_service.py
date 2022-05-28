import pandas as pd

import pickle


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from apscheduler.schedulers.background import BackgroundScheduler

def retrain_model():
    dataset = pd.read_csv('dataset.csv')

    y = dataset['weather_condition']
    x = dataset.drop(['weather_condition'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x,y,train_test_split=0.8, shuffle=True)
    gnb = GaussianNB()
    gnb.fit(x,y)
    current_accuracy = accuracy_score(gnb.predict(X_test))

    if os.path.exists('nb_model.pickle','rb'):
        prev_model_file = open('nb_model.pickle','rb')
        prev_model = pickle.load(prev_model_file)
        prev_model_file.close()

        prev_accuracy = accuracy_score(prev_model.predict(X_test), y_test)
    else:
        prev_accuracy = -1

    if current_accuracy > prev_accuracy:
        model_file = open('nb_model.pickle', 'wb')
        pickle.dump(gnb, model_file)
        model_file.close()