import pandas as pd

data = pd.read_csv(
    r'C:\Users\slshp\PycharmProjects\ML_Python\Linear_Regression\Boston.csv')
y_data = data['MEDV']
x_data = data.drop('MEDV', axis=1).values

from sklearn import preprocessing as spp
from sklearn.model_selection import train_test_split
scaler = spp.MinMaxScaler()

x_pre_data = scaler.fit_transform(x_data)
model_data = train_test_split(x_pre_data,
                              y_data,
                              test_size=0.1,
                              random_state=42)
print('Boston data is done!')