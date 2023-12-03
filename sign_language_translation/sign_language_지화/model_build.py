from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, RNN, SimpleRNN, MaxPool1D, Dropout, Flatten, Conv1D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
from functions import *
from data_process import *

#tensorboard / train 폴더 tensorboard --logdir=.
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# model train(rnn)
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, activation='relu', input_shape=(10, 63)))
model.add(SimpleRNN(128, return_sequences=True, activation='relu'))
model.add(SimpleRNN(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = './model/{epoch:02d}_{categorical_accuracy:4f}.h5'
model_checkpoint_callback = ModelCheckpoint(filepath = model_path, monitor='categorical_accuracy', verbose=1, save_best_only = True, save_weights_only=False, mode= 'max')

model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[tb_callback, model_checkpoint_callback])

model.summary()

model.save('./model/action.h5')

# #cnn
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=(10, 63)))
# model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(MaxPool1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(MaxPool1D(pool_size=2))
# model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(MaxPool1D(pool_size=2))
# model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"))
# model.add(Dropout(rate=0.2))
# model.add(Flatten())
# model.add(Dense(512, activation="relu"))
# model.add(Dense(actions.shape[0], activation="softmax"))
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
# model_dir = './model'
# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)
#
# model_path = './model/{epoch:02d}_{categorical_accuracy:4f}.h5'
# model_checkpoint_callback = ModelCheckpoint(filepath = model_path, monitor='categorical_accuracy', verbose=1, save_best_only = True, save_weights_only=False, mode= 'max')
#
# model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test) , callbacks=[tb_callback, model_checkpoint_callback])
#
# model.summary()
#
# model.save('./model/action.h5')





