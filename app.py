#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template, request


# In[4]:


import joblib


# In[5]:


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":  
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        income = float(income)
        age = float(age)
        loan = float(loan)
        print(income, age, loan)
        
        modeldt = joblib.load("CCU_DT")
        preddt = modeldt.predict([[income, age, loan]])
        s1 = "The possibility of default base on decision tree is " + str(preddt)
        modelgb = joblib.load("CCU_GB")
        predgb = modelgb.predict([[income, age, loan]])
        s2 = "The possibility of default base on gradient boosting is " + str(predgb)
        modellr = joblib.load("CCU_LR")
        predlr = modellr.predict([[income, age, loan]])
        s3 = "The possibility of default base on Linear Regression is " + str(predlr)
        modelnn = joblib.load("CCU_NN")
        prednn = modelnn.predict([[income, age, loan]])
        s4 = "The possibility of default base on Neural Network is " + str(prednn)
        modelrf = joblib.load("CCU_RF")
        predrf = modelrf.predict([[income, age, loan]])
        s5 = "The possibility of default base on Random Forest is " + str(predrf)      

        return(render_template("index.html", result1=s1 ,result2=s2 ,result3=s3 ,result4=s4 ,result5=s5))
    else:
        return(render_template("index.html", result1="2",result2="2",result3="2",result4="2",result5="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




