import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.utils import np_utils
from keras.utils import to_categorical
import pandas as pd
import pymysql

mysql_connection = pymysql.connect(host='your-database-hostname',
                    user='your-database-username',
                    password='your-database-pass',
                    db='your-database-name',
                    charset='utf8',
                    cursorclass=pymysql.cursors.DictCursor)
                    
sql = "SELECT * FROM `student_data`"
data = pd.read_sql(sql, mysql_connection)

x = np.array(data[['gre', 'gpa', 'rank']].values.tolist())
y = np.array(data['admit'].values.tolist())
y = to_categorical(y)

model = Sequential()

model.add(Dense(32, input_shape=(3,)))
model.add(Activation('relu'))
model.add(Dense(64, input_shape=(3,)))
model.add(Activation('relu'))
model.add(Dense(128, input_shape=(3,)))
model.add(Activation('softmax'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, epochs=1000, batch_size=100, verbose=0)
score = model.evaluate(x, y)
print("\n Training Accuracy:", score[1])
model.summary()
