import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__) ## Starting point of my app from where the application will run

## Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/') ## It says I should definitely go to home page
def home():
    # Return html page for home page. Render_template function will look for a template folder
    return render_template('home.html') 

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']  ##Whatever data provided as input will be given in the form of json format and stored in data variable
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_Data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_Data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    print(output)
    return render_template("home.html", prediction_text = "The House price prediction is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)