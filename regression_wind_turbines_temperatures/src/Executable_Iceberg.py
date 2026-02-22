#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Connection PI
import PIconnect as PI
import xlwings as xw
import clr
from System.Net import NetworkCredential
from PIconnect.PIConsts import AuthenticationMode
from PIconnect.PIConsts import UpdateMode, BufferMode
from datetime import *


# In[11]:


import sys
import os
# Load libraries
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle

import warnings
warnings.filterwarnings('ignore')


# In[12]:


#LOGUEO
def logueo():
    global u,p
    #u = input('>> Ingrese usuario\n')
    #p = getpass.getpass('>> Ingrese contraseña\n')
    try:
        if(len(u)>0):
            print("--Logueado--")
            
    except:
        df = pd.read_csv("./input/Credentials.txt", sep=";")
        u = df['u'].values[0]
        p = df['p'].values[0]


# In[13]:


#FILTRO DE VALORES BAD
def filterarr(arr1):
    arr2=[]

    for e in arr1:
        try:
            arr2.append(float(e))
        except:    
            arr2.append(None)
    return arr2


# In[24]:


#LECTURA DE ARCHIVO
def lectura_input(argument):
    
    #Lectura de tags
    df_input = pd.read_csv("./input/INV_INPUT.csv",sep=";")
    df_tags = pd.DataFrame()

    for i in df_input.columns[1:]:

        df_var = pd.DataFrame()
        df_var["TAG"] = df_input[i]

        df_var["DESCRIPTION"] = df_input["ID_WTG"].astype(str) + " - " + i

        df_tags = pd.concat([df_tags,df_var])

    df_tags = df_tags.reset_index(drop=True)
    
    date = pd.read_csv("./input/INV_INPUT_PARAMETERS.csv",sep=";")
    server = date["SERVER"].values[0]

    if(argument == "manual"):

        print("Modo manual")
        
        d_start = date["START"].values[0]
        d_start = datetime.strptime(d_start, '%d/%m/%Y')
        d_start = d_start.strftime("%Y-%m-%d %H:%M:%S")

        d_end = date["END"].values[0]
        d_end = datetime.strptime(d_end, '%d/%m/%Y')
        d_end = d_end.strftime("%Y-%m-%d %H:%M:%S")

    else: 

        d_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        d_end = (datetime.now() - timedelta(days = 7)).strftime("%Y-%m-%d %H:%M:%S")
    
    
    return df_tags, d_start, d_end, server


# In[25]:


#DESCARGA DE DATOS DE PI

def descarga_PI(tag_fuente,d_start, d_end, server):
    
    e = tag_fuente
    
    #Proceso de descarga de datos
    if(d_start == d_end):
        print(">> Los datos horario ya están actualizados")
    else:

        PI.PIConfig.DEFAULT_TIMEZONE = 'Etc/GMT+5'
        print(datetime.today())

        df = pd.DataFrame()

        print(PI.PIConfig.DEFAULT_TIMEZONE)

        print(">> Ejecutando como modo externo ")
        logueo()
        with PI.PIServer(server=str(server),username=u, password=p, authentication_mode=AuthenticationMode.WINDOWS_AUTHENTICATION) as server:

            print(e)
            points = server.search(e)
            data = points[0].interpolated_values(d_start, d_end, '1h')
            print(points[0])

            df["Time"]=data.index
            df[str(e)]=filterarr(data.values)

        df=df.set_index('Time')        

        df.reset_index(level=0, inplace=True)

        df['Time'] = pd.to_datetime(df['Time'])

        df=df.set_index('Time')

    return df


# In[31]:


#SE OBTIENEN DATOS

def obtencion_datos_entrada(argument):
    
    df_tags, d_start, d_end, server = lectura_input(argument)
    
    df_final = pd.DataFrame()
    i=0
    error = 0

    while(i<len(df_tags)):

        print("Tag numero: " + str(i))
        df = descarga_PI(df_tags["TAG"].to_list()[i], d_start, d_end, server)
        df_r = df.copy()
        df_r.columns = [df_tags["DESCRIPTION"].to_list()[i]]
        df_final = pd.concat([df_final,df_r],axis=1)
        i+=1
        error = 0
    
    df_final.index = df_final.index.tz_localize(None)
    df_final.reset_index(inplace=True)
    
    return df_final


# In[33]:


#PROCESAMIENTO

def procesamiento(names,dataset):

    df_format_predict = pd.DataFrame()

    for name in names:
        cols = []
        for i in dataset.columns:
            if (("Time" in i)|(name in i)):
                cols.append(i)
        df_format = dataset[cols]
        df_format = df_format.melt(id_vars='Time', var_name='Variable', value_name=name)
        df_format[["WTG","Variable"]] = df_format["Variable"].str.split("-",expand=True)
        df_format = df_format.drop("Variable",axis=1)

        if(len(df_format_predict)>0):
            df_format_predict = pd.merge(df_format_predict, df_format, on=['Time','WTG'], how='inner')
        else:
            df_format_predict = df_format.copy()

    return df_format_predict


# In[47]:


#APLICACION DE MODELO

def aplicacion_modelo(col_input,col_ouput,df_format_predict):
    
    output_dir = "./models - CART/"
    loaded_transformers = []

    col_extract = df_format_predict.pop("WTG")
    df_format_predict.insert(1,"WTG",col_extract)
    df_format_predict = df_format_predict.dropna(subset=df_format_predict.columns[2:6])
    data_test = df_format_predict[col_input].copy()

    for i,name in enumerate(col_ouput):

        print(name)

        try:

            modelo_pkl_path = output_dir + name + ".pkl"

            with open(modelo_pkl_path, 'rb') as f:
                modelo_cargado = pickle.load(f)

            df_format_predict[name+"_model"] = modelo_cargado.predict(data_test)

            with open(os.path.join(output_dir, f'transformerYJ_{name}.pkl'), 'rb') as transformer_file:
                loaded_transformer = pickle.load(transformer_file)

            df_format_predict[name+"_model"] = loaded_transformer.inverse_transform(df_format_predict[name+"_model"].values.reshape(-1, 1)).flatten()
            
            print(f">> Listo")
        except Exception as e:
            print(f">> No se encontro el siguiente modelo o transformer YJ: {e}")
        
    return df_format_predict


# In[68]:


#AJUSTE SIGMA DE INDICADOR

def indicador_salud(col_names,df_format_predict):
        
    for i in col_names[5:]:
        
        try:
            
            #Sigma o factor de ajuste
            
            if ("Transformer" in i):
                df_format_predict[i + "_sigma"] = df_format_predict[i+"_model"]*1.08 #1.07
            elif (("Generator" in i) | ("Slow shaft" in i)):
                df_format_predict[i + "_sigma"] = df_format_predict[i+"_model"]*1.07 #1.055
            elif (("Gearbox" in i) | ("Hydraulic oil" in i)):
                df_format_predict[i + "_sigma"] = df_format_predict[i+"_model"]*1.068 #MAE*2.5
            elif ("Exchanger" in i):
                df_format_predict[i + "_sigma"] = df_format_predict[i+"_model"]*1.06

            #Indicador de sobretemperatura
            df_format_predict[ i + "_HI"] = 0
            p75 = df_format_predict[i].quantile(0.75)
            df_format_predict.loc[(df_format_predict[i] > df_format_predict[i+"_sigma"]) & (df_format_predict[i]>p75), i + "_HI"] = 1

        except Exception as e:
            
            print(f">> No se encontró variable: {e}")
            continue
        
    return df_format_predict


# In[69]:


# #TRF
# gearbox solo 1
# #Radiador
# 2 se;ales


# In[44]:


if __name__ == "__main__":
    
    if len(sys.argv)>1:
        
        print(">> Argumento:",sys.argv[1])
        argument = sys.argv[1] 
    
    # try:
    
    dataset = obtencion_datos_entrada(argument)
                     
    col_names = [
    "Active Power",
    "Wind direction",
    "Wind Speed",
    "Ambient temperature",
    "Nacelle ambient temperature",
    "Slow shaft front bearing temperature",
    "Slow shaft rear bearing temperature",
    "Gearbox oil temperature",
    "Gearbox bearing temperature",
    "Hydraulic oil temperature",
    "Top Heat Exchanger Temperature", #radiador superior
    "Bottom Heat Exchanger Temperature", #radiador inferior
    "Generator winding temperature U", #3,15 a 5.9 MW
    "Generator winding temperature V",
    "Generator winding temperature W",
    "Generator slip ring temperature",
    "Generator DE bearing temperature",
    "Generator NDE bearing temperature",
    "Transformer temperature phase A",
    "Transformer temperature phase B",
    "Transformer temperature phase C",
    "Generator Cooling Water Temperature" #Refrigerante CNV
    ]
    
    col_input = ["Active Power","Wind Speed","Ambient temperature","Nacelle ambient temperature",]
    col_output = col_names[5:]

    df_format_predict = procesamiento(col_names,dataset)
    df_format_predict = aplicacion_modelo(col_input,col_output,df_format_predict)

    df_format_predict = indicador_salud(col_names,df_format_predict)
    df_format_predict.to_csv("./output/Resultado.csv", index=False)
    print(">> PROCESO COMPLETADO")



