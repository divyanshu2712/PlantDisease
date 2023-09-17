from flask import Flask,render_template,request,abort,redirect
from flask_dropzone import Dropzone
from data import classes,cure,cure_link
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

image=None
access=False

app = Flask(__name__)
app.config.update(DROPZONE_TIMEOUT=6*5*1000)
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_MAX_FILE_EXCEED']=1
app.config['DROPZONE_DEFAULT_MESSAGE']="Place Your Plant Image Here"
app.config['DROPZONE_MAX_FILE_SIZE']=80

dropzone=Dropzone(app)


@app.route('/predict',methods=['GET','POST'])
def predict():
    global image
    global access
    f=request.files.get('file')
    if f:
        image=np.array(Image.open(BytesIO(f.read())))
        image=cv2.resize(image,(256,256))
        image=np.expand_dims(image,axis=0)
        # pred=model.predict(image)
        # class_pred=classes[np.argmax(pred)]
        # confidence=round(np.max(pred)*100,2)
        access=True
    return render_template("predict.html")
@app.route('/result',methods=["POST"])
def result():
    global image
    a=request.form.get("d_name")
    print(a)
    model=tf.keras.models.load_model(f"models/{a}")
    pred=model.predict(image)
    class_pred=classes[a][np.argmax(pred)]
    confidence=round(np.max(pred)*100,2)
    if confidence!=None:
        if(class_pred=="healthy"):
            healthy="Healthy"
            disease=False
            return render_template("result.html",healthy=healthy,disease=disease)
        else:
            healthy="Not Healthy"
            disease=True
            disease_cure=cure[a][class_pred]
            disease_cure_links=cure_link[a][class_pred]
            return render_template("result.html",healthy=healthy,disease=disease,disease_cure=disease_cure,disease_cure_links=disease_cure_links,class_pred=class_pred,confidence=confidence)
    else:
        abort(403)

@app.route("/about")
def aboutus():
    global confidence
    confidence=None
    return render_template("aboutus.html")

@app.route("/")
def index():
    global confidence
    confidence=None
    return render_template("index.html")
