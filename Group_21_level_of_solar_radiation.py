import shutil
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st


train_data = pd.read_csv('SolarPrediction.csv',index_col=0)

train_data = train_data.reindex(np.random.permutation(train_data.index))

def process_features(dataset):
    
    features_to_use = dataset.loc[:,['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed']]
    ready_to_use = features_to_use.copy()
   
    return ready_to_use

def process_single_prediction(s):
    g_f=dict()   
     
    g_f['Temperature'] = s[0]
    g_f['Pressure'] = s[1]
    g_f['Humidity'] = s[2]
    g_f['WindDirection(Degrees)'] = s[3]
    g_f['Speed'] = s[4]
                   
    g_f=pd.DataFrame(data=[g_f.values()],columns=['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed'])    
    
    g_f.loc[1,['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed']]=train_data.loc[1479722402,['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed']]
   
    return g_f

train_dataset = train_data.sample(frac=0.8, random_state= 121)
test_dataset = train_data.drop(train_dataset.index)

train_features = process_features(train_dataset)
test_features = process_features(test_dataset)

train_labels = train_dataset['Radiation']
test_labels = test_dataset['Radiation']

feature_cols = [tf.feature_column.numeric_column(my_feature)for my_feature in train_features]

def model_input(option,batch_size=0,num_epochs=None, shuffle=True):
    
    if option == 'train':
        return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x = train_features,
        y = train_labels,
        
        batch_size = batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
        )
    
    if option == 'evaluate':
        return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x = test_features,
        y = test_labels,
       
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
        )
    
def prediction_input(data,num_epochs=None,shuffle=True):
   
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = data,
    num_epochs=num_epochs,
    shuffle=shuffle
    )


radiation_linear_estimator = tf.estimator.LinearRegressor(
     feature_columns=feature_cols,
)

radiation_linear_estimator.train(input_fn=model_input('train',batch_size=10,num_epochs=None), steps=500)

inputFn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
tf.feature_column.make_parse_example_spec(feature_cols))

OUTDIR = 'rad_linear regression'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
modelBasePath = os.path.join(OUTDIR, "model")
modelPath = radiation_linear_estimator.export_saved_model(modelBasePath, inputFn)

st.title('Level Of Solar Radiation')

#'Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed'

Temperature =int(st.number_input('Temperature in degrees Fahrenheit:',max_value = 108.7))

Pressure=float(st.number_input('Barometric pressure in Hg:',min_value = 25,max_value = 31))

Humidity = int(st.number_input('Humidity percent:',min_value = 1))

WindDirection_Degrees = float(st.number_input('Wind direction in degrees:',min_value=0.00,max_value=360.00))

Speed = float(st.number_input('Wind speed in km per hour:',min_value=0.00))


savedModelPath = modelPath
importedModel = tf.saved_model.load(savedModelPath)

def predict(dfeval, importedModel):
    colNames = dfeval.columns
    dtypes = dfeval.dtypes
    predictions = list()
    for row in dfeval.iterrows():
        example = tf.train.Example()
        for i in range(len(colNames)):
          dtype = dtypes[i]
          colName = colNames[i]
          value = row[1][colName]
          if dtype == "object":
            value = bytes(value, "utf-8")
            example.features.feature[colName].bytes_list.value.extend(
                [value])
          elif dtype == "float":
            example.features.feature[colName].float_list.value.extend(
                [value])
          elif dtype == "int":
            example.features.feature[colName].int64_list.value.extend(
                [value])

        predictions.append(
          importedModel.signatures["predict"](
            examples=tf.constant([example.SerializeToString()])))
    return predictions

    

if st.button('Determine'):
    
    x=predict(process_single_prediction([Temperature,Pressure,Humidity,WindDirection_Degrees,Speed]),importedModel)

    z = float(x[0]['predictions'][0][0])
    st.success('Solar Level Radiation in watts per sq meter:')
    st.header(z)
    