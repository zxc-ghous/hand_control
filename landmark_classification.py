import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns



cols=['id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
       '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
       '34', '35', '36', '37', '38', '39', '40', '41']
dataset=pd.read_csv(r'hand_landmarks.csv',names=cols, header=None)

y_dataset = dataset['id']
X_dataset = dataset.loc[:,'0':]

NUM_CLASSES = len(y_dataset.value_counts()) # 2 класса


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)


input_tensor = layers.Input(shape=(42,))
x = layers.Dropout(0.2)(input_tensor)
x = layers.Dense(20, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(10, activation='relu')(x)
output_tensor = layers.Dense(1, activation='sigmoid')(x)

model = Model(input_tensor, output_tensor)
print(model.summary())

callbacks=[
       callbacks.EarlyStopping(patience=50,verbose=1,monitor='val_loss')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=600,
    batch_size=128,
    validation_data=(X_test, y_test),
)

sns.lineplot(x=range(len(history.history['loss'])),y=history.history['loss'],label='loss')
sns.lineplot(x=range(len(history.history['val_loss'])),y=history.history['val_loss'],label='val_loss')
sns.lineplot(x=range(len(history.history['accuracy'])),y=history.history['accuracy'],label='accuracy')
sns.lineplot(x=range(len(history.history['val_accuracy'])),y=history.history['val_accuracy'],label='val_accuracy')
plt.show()

model_save_path=r'saved_models\model_600'

model.save(model_save_path)