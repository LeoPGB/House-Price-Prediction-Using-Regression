import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('/content/house_data.csv')

df['year'] = df['date'].str[:4].astype(int)
df['house_age'] = np.NaN

df.head()

for i,j in enumerate(df['yr_renovated']):
    if (j==0):
        df['house_age'][i] = df['year'][i] - df['yr_built'][i]
    else:
        df['house_age'][i] = df['year'][i] - df['yr_renovated'][i]
print('done')

df.head()

df.drop(['date','yr_built','yr_renovated','year','id','lat','zipcode','long'], axis=1, inplace=True)

df.head()

df.describe()

df = df[df['house_age']!=-1]

df.describe()

df.head()

df.shape

df.count()

df.dtypes

#plot
for i in df.columns:
    sns.displot(df[i])
    plt.show()

#pair plot
plt.figure()
sns.pairplot(df)
plt.show()

#heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)
plt.show()

#boxplot
for i in df.columns:
    sns.boxplot(x=df[i])
    plt.show()

#split data into in/output
x = df.drop('price', axis=1)
y = df['price']

x.head()

y.head()

print(x.shape)
print(y.shape)

#model
model = keras.Sequential()
model.add(layers.Dense(14,activation='relu'))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='mse',optimizer='adam', metrics=['mse','mae'])
hist = model.fit(x,y,validation_split=0.33,batch_size=32,epochs=25)
#(loss=training loss, val_loss=validation loss)

model.summary()

#training plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'],loc='upper right')
plt.show()

#prediction
x_pred = np.array([[2,3,1280,5550,1,0,0,4,7,2280,0,1440,5750,60]])
x_pred = np.array(x_pred,dtype=np.float64)
y_pred = model.predict(x_pred)
print('x = ', x_pred, '\ny = ', y_pred[0])

y.head()

x_pred = np.array([[3,1,1180,5650,1,0,0,3,7,1180,0,1340,5650,59.0]])
x_pred = np.array(x_pred,dtype=np.float64)
y_pred = model.predict(x_pred)
print('x = ', x_pred[0], '\ny = ', y_pred[0][0])
