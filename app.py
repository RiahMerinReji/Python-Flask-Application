from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

data = pd.read_csv('Social_Network_Ads.csv')


with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    Age = float(request.form['age'])
    EstimatedSalary = float(request.form['estimatedSalary'])

    features=np.array([[Age,EstimatedSalary]])
    prediction = model.predict(features)
    target =  prediction[0]
    

    if target == 1:
        output = "Customer Purchased"
    else:
        output = "Customer Not Purchased"
      
    return render_template('index.html',check_purchase = output)
   

if __name__=='__main__':
    app.run(debug=True)