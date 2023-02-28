from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors=CORS(app)
model=pickle.load(open('RegressorModel.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    spx=request.form.get('spx')
    uso=request.form.get('uso')
    slv=request.form.get('slv')
    eurusd=request.form.get('eur/usd')

    
    prediction=model.predict(pd.DataFrame(columns=['SPX', 'USO', 'SLV', 'EUR/USD'],
        data=np.array([spx,uso,slv,eurusd]).reshape(1, 4)))
    print(prediction)
    
    

    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)