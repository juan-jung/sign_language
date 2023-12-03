#eval
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from functions import *
from tensorflow.keras.models import load_model
from data_process import *

model = load_model('./model/cnn.h5')


yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))