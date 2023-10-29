#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, jsonify, request
import pickle
import pandas as pd


# In[7]:


app = Flask(__name__)


# In[8]:


@app.route('/',methods= ['GET','POST'])
def home():
    if(request.method == 'GET'):
        
        data = 'Hello World'
        return jsonify({'data':data})
        


# In[16]:


@app.route('/predict/')
def price_predict():
    model = pickle.load(open('model.pickle','rb'))
    area = request.args.get('area')
    bedrooms = request.args.get('bedrooms')              
    bathrooms = request.args.get('bathrooms')
    stories = request.args.get('stories')
    parking = request.args.get('parking')

    
    test_df = pd.DataFrame({'area' :[area],
                        'bedrooms' :[bedrooms],
                        'bathrooms' :[bathrooms],
                        'stories' :[stories],
                        'parking' :[parking]
                       })
    
    pred_price = model.predict(test_df)
    
    return jsonify({'House Price ' : str(pred_price)})


# In[ ]:





# In[11]:


if __name__ == '__main__':
    app.run(debug = True)


# In[ ]:




