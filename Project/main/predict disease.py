#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# loading model
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))

#Unpickle dataframes
symptom_severity = pd.read_pickle("symptom_severity.pkl")
disease_description = pd.read_pickle("disease_description.pkl")
disease_precaution = pd.read_pickle("disease_precaution.pkl")

def DiseasePrediction(text):

    bool_list = []
    
    text_token_list = text.split()
    text_token_list = [ps.stem(w) for w in text_token_list]
    text_token_set = set(text_token_list)

    for ele in symptom_severity["Symptom"]:
        ele = ele.split('_')
        ele = [ps.stem(w) for w in ele]
        ele_set = set(ele)

        flag=0

        for i in range(len(text_token_list)):
            if(len(ele) > 1):
                if(ele_set.issubset(text_token_set)):
                    flag = 1
                else:
                    flag = 0
            else:
                if(text_token_list[i] == ele[0]):
                    flag = 1
                else:
                    flag = 0

        bool_list.append(flag)

    symptom_severity["bool_list"] = bool_list
    symptom_severity["weight_bool_list"] = symptom_severity["weight"] * symptom_severity["bool_list"]
    weight_bool_list = list(symptom_severity["weight_bool_list"])

    # Disease Prediction
    X_arr = np.array(weight_bool_list)
    y_pred = loaded_model.predict(X_arr.reshape(1, -1))
    Predicted_Disease = y_pred[0]
    print("WE ARE PREDICTING THAT YOU HAVE ------- ",Predicted_Disease.upper())

    #Disease Description
    for i in range(len(disease_description)):
        if(disease_description["Disease"][i] == Predicted_Disease):
            print("DESCRIPTION OF THE DISEASE : ---- \n")
            print(disease_description["Description"][i])

    # Disease Precaution
    for i in range(len(disease_precaution)):
        if(disease_precaution["Disease"][i] == Predicted_Disease):
            print("PRECAUTUIONs OF THE DISEASE : ---- \n")
            print("PRECAUTUION 1")
            print(disease_precaution["Precaution_1"][i])
            print("################# \n")
            print("PRECAUTUION 2")
            print(disease_precaution["Precaution_2"][i])
            print("################# \n")
            print("PRECAUTUION 3")
            print(disease_precaution["Precaution_3"][i])
            print("################# \n")
            print("PRECAUTUION 4")
            print(disease_precaution["Precaution_4"][i])
            print("################# \n")
            
symptoms_by_user = input("Write about your SYMPTOMS: ")
DiseasePrediction(symptoms_by_user)


# In[ ]:





# In[118]:





# In[ ]:





# In[ ]:




