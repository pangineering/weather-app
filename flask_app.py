from flask import Flask, request, render_template
from Utilities.model_service import retrain_model

import time
import json
import pickle

app = Flask(__name__)

model_file = open('nb_model.pickle','rb')
model = pickle.load(model_file)
model_file.close()

@app.route('/',methods=['GET'])
def show_predict_page():
    return render_template('predict.html')

@app.route('/',methods=['POST'])
def show_predict_result():
    temperature = request.form['temperature']
    humidity = request.form['humidity']
    wind_direction = request.form['wind_direction']
    wind_speed = request.form['wind_speed']
    pressure = request.form['pressure']

    model_file = open('nb_model.pickle','rb')
    model = pickle.load(model_file)
    model_file.close()

    inference_result = model.predict([[pressure,wind_speed,humidity,temperature,wind_direction]]).tolist()

    return render_template('predict.html',result=inference_result)
    #return json.dumps(inference_result)

@app.route('/<name>')
def hello_name(name):
    return 'Hello' + name

@app.route('/add/<int:input_num>')
def add_number(input_num: int):
    return str(input_num + 1)

#def print_date_time():
#    print(time.strftime("%A,%d.%B %Y %I:%M:%S %p"))
if __name__ == '__main__':
    schedule = BackgroundScheduler(daemon=True)
    schedule.add_job(retrain_model,trigger='cron',hour=3)
    schedule.start()
    app.run()

    