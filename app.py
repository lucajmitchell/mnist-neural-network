from flask import Flask, render_template, request, redirect
from neuralNetworkUtils import getData



app = Flask(__name__)
trainingLabels, trainingImages, testingLabels, testingImages = getData("data/mnist_train.csv", "data/mnist_test.csv")



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/getImageData")
def getImageData():

    imageData = {}
    images = testingImages.T.tolist()[:1000] # Only load first 1000 images for viewing to reduce load times
    labels = testingLabels.tolist()

    for i, pixels in enumerate(images):
        imageData[i] = {
            "index": i,
            "pixels": pixels,
            "actual": labels[i],
            "predicted": "N/A"
        }

    return imageData



app.run(debug=True)