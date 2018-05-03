import numpy as np
import keras.models
from keras.models import model_from_json
from imageio import imread
from scipy.misc import  imresize,imshow

json_file = open('EMmodel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("EMmodel.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
x = imread('a.png',pilmode='L')
x = np.invert(x)

x = imresize(x,(28,28))
#imshow(x)
x = x.reshape(1,28,28,1)
output = 0
out = loaded_model.predict(x)
print(out.argmax(axis=1)[0])
id_val = max(out[0])
for i in range(27):
    if out[0][i] == id_val:
        output = i

print('Value is = ', output)
print("List MAX = ", max(out[0]))
print(out)
print(np.argmax(out,axis=1))

print(chr(78))
