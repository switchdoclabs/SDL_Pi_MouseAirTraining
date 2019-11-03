#import libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image

print("import complete")
# load model 

img_width = 150
img_height = 150

class_names = ["Cat", "NotCat"]

model = tf.keras.models.load_model("CatNotCat.trained",compile=True)
print (model.summary())

# do single image
imageName = "ukbench09941.jpg"

testImg = Image.open(imageName)
testImg.load()

testImg = testImg.resize((150, 150), Image.ANTIALIAS)
data = np.asarray( testImg, dtype="float" )

data = np.expand_dims(data, axis=0)
singlePrediction = model.predict(data, steps=1)

NumberElement = singlePrediction.argmax()
Element = np.amax(singlePrediction)
print(NumberElement)
print(Element)
print(singlePrediction)

print ("Our Network has concluded that the file '"
        +imageName+"' is a "+class_names[NumberElement])
print (str(int(Element*100)) + "% Confidence Level")

# do single image
imageName = "cat.1332.jpg"

testImg = Image.open(imageName)
testImg.load()

testImg = testImg.resize((150, 150), Image.ANTIALIAS)
data = np.asarray( testImg, dtype="float" )

data = np.expand_dims(data, axis=0)
singlePrediction = model.predict(data, steps=1)

NumberElement = singlePrediction.argmax()
Element = np.amax(singlePrediction)
print(NumberElement)
print(Element)
print(singlePrediction)

print ("Our Network has concluded that the file '"
        +imageName+"' is a "+class_names[NumberElement])
print (str(int(Element*100)) + "% Confidence Level")

# classify the input image
(notcat, cat) = model.predict(data)[0]
print (notcat, cat)

