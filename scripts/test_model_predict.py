import json, joblib, os, pandas as pd

data=json.load(open('autonomous_result_iris.json'))
model_path=data.get('optimization_report',{}).get('artifacts',{}).get('model')
print('model_path:', model_path)
print('exists:', os.path.exists(model_path))
if model_path and os.path.exists(model_path):
    m=joblib.load(model_path)
    print('model type:', type(m))
    df=pd.DataFrame([{'sepal length (cm)':5.1,'sepal width (cm)':3.5,'petal length (cm)':1.4,'petal width (cm)':0.2}])
    try:
        pred=m.predict(df)
        print('prediction:', pred)
    except Exception as e:
        print('prediction error:', e)
else:
    print('model not found')
