#!/usr/bin/env python
# coding: utf-8

# #  Import library for read the dataset
# 

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("AAPL.csv")


# In[3]:


df


# # Data Analysis

# In[4]:


df.info()


# In[5]:


print(df.describe)


# In[6]:


print(df.shape)


# In[7]:


print(df.columns)


# In[8]:


print(df.dtypes)


# In[9]:


df.count()


# In[10]:


df.nunique()


# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[13]:


df.head(10)


# In[14]:


df1 = df[('close')]


# In[15]:


df1


# In[29]:


df2 = df[('open')]


# In[30]:


df2


# In[35]:


df3 = df[(df['high'] <= 200) & (df['low'] <= 128.91)][['high', 'low']]


# In[36]:


df3


# # Visualization

# In[28]:


import matplotlib.pyplot as plt
import numpy as np


# In[27]:


plt.plot(df1)


# In[45]:


condition_df2 = df['open'] <= 130.58  # Replace 'column_name' with the actual column name you want to filter

# Set the condition for df1
condition_df1 = df['close'] <= 132.045  # Replace 'column_name' with the actual column name you want to filter

# Plot the points from df2 and df1 based on the conditions
plt.plot(df2[condition_df2], label='open')
plt.plot(df1[condition_df1], label='close')

# Add labels, title, and legend
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('open and close Data graph ')
plt.legend()

# Show the plot
plt.show()


# In[47]:


# Plotting the histograms
df.plot(kind='hist', bins=10, alpha=0.7, title='Histogram of Close, High, Low, and Open Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[49]:


# Extract the columns you want to plot
columns_to_plot = ['close', 'high', 'low', 'open',
       'volume', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']

# Plot histograms for each column
for column in columns_to_plot:
    plt.figure()
    plt.hist(df[column], bins=20, alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()


# # Machine Learning model

# In[52]:


import numpy as np


# In[53]:


df1


# In[55]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[56]:


print(df1)


# In[57]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.75)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[58]:


test_size, training_size


# In[59]:


test_data


# In[60]:


train_data


# In[61]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[62]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)


# In[63]:


print(X_test.shape), print(ytest.shape)


# In[64]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[87]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# Create the Stacked LSTM model
model = Sequential()

# First LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# Second LSTM layer
model.add(LSTM(units=50, return_sequences=True))
# Third LSTM layer
model.add(LSTM(units=50))
# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[88]:


model.summary()


# In[90]:


# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Make predictions
predicted_values = model.predict(X_test)


# In[91]:


import tensorflow as tf


# In[92]:


tf.__version__


# In[93]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[94]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[95]:


import math
from sklearn.metrics import mean_squared_error

# Assuming you have defined train_predict and y_train as your predicted and actual training values
rmse_train = math.sqrt(mean_squared_error(y_train, train_predict))
print(f"Train RMSE: {rmse_train}")


# In[97]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[98]:


# Inverse scale the predicted values and actual test values
predicted_values = scaler.inverse_transform(predicted_values)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[100]:


print(predicted_values)


# In[101]:


print(actual_values)


# In[102]:


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Prices')
plt.plot(predicted_values, label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[104]:


len(test_data)


# In[109]:


x_input=test_data[215:].reshape(1,-1)
x_input.shape


# In[110]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[111]:


temp_input


# In[112]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[113]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[114]:


len(df1)


# In[115]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[117]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[118]:


df3=scaler.inverse_transform(df3).tolist()


# In[119]:


plt.plot(df3)


# In[ ]:




