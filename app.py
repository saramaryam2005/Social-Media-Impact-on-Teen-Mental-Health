import pickle
from flask import Flask,request,app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app= Flask (__name__)
lr_model=pickle.load(open("lr_model (1).pkl","rb"))
scalar = pickle.load(open("Min_Maxscaler.pkl", "rb"))
@app.route("/")
def home():
    return render_template("home.html") 

@app.route("/predict_api", methods=["post"])

def predict_api():
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) 
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=lr_model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route ("/predict", methods=["post"])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=lr_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The chances of Heart Disease is (0=no,1=yes): {}".format(output))


if __name__ =="__main__":
    app.run(debug=True)