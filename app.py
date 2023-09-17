from flask import Flask,render_template,request,abort,redirect
from flask_dropzone import Dropzone
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2


model=tf.keras.models.load_model("models/Potato")
classes=["EarlyBlight","LateBlight","Healthy"]
class_pred=""
confidence=None
access=False
cure={
    'EarlyBlight':["Prune or stake plants to improve air circulation and reduce fungal problems.",
"Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut.",
"Keep the soil under plants clean and free of garden debris. Add a layer of organic compost to prevent the spores from splashing back up onto vegetation.",
"Drip irrigation and soaker hoses can be used to help keep the foliage dry.",
"For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed.",
"Containing copper and pyrethrins, Bonide® Garden Dust is a safe, one-step control for many insect attacks and fungal problems. For best results, cover both the tops and undersides of leaves with a thin uniform film or dust. Depending on foliage density, 10 oz will cover 625 sq ft. Repeat applications every 7-10 days, as needed.",
"SERENADE Garden is a broad spectrum, preventative bio-fungicide recommended for the control or suppression of many important plant diseases. For best results, treat prior to foliar disease development or at the first sign of infection. Repeat at 7-day intervals or as needed.",
"Remove and destroy all garden debris after harvest and practice crop rotation the following year.",
"Burn or bag infected plant parts. Do NOT compost."],

    "LateBlight":[
    "Remove Infected Plant Parts: As soon as you notice signs of late blight (brown lesions on leaves, dark spots on fruit), remove and destroy the infected plant parts. Do not compost infected material, as the spores can survive composting.",
    
    "Prune for Air Circulation: Prune the plants to improve air circulation. This helps reduce humidity around the plants, making it less favorable for the disease to spread.",
    
    "Water at the Base: Water the plants at the base using drip irrigation or a soaker hose to keep the foliage dry. Avoid overhead watering, as wet foliage can promote the spread of late blight.",
    
    "Apply Fungicides: Consider using fungicides approved for late blight control. Copper-based fungicides and products containing chlorothalonil or mancozeb can be effective. Follow the manufacturer's instructions for application rates and timing.",
    
    "Use Resistant Varieties: If possible, choose tomato or potato varieties that are resistant to late blight. These varieties are less susceptible to the disease.",
    
    "Apply Fungicides Preventatively: Fungicides are most effective when applied preventatively, especially during periods of high humidity or wet weather. Begin applying fungicides before late blight symptoms appear and continue at regular intervals as recommended on the product label.",
    
    "Rotate Crops: Practice crop rotation by not planting tomatoes or potatoes in the same location for consecutive years. This can help reduce the buildup of late blight spores in the soil.",
    
    "Clean Garden Debris: After the growing season, clean up and remove all garden debris, including fallen leaves and fruit, to reduce overwintering spores.",
    
    "Monitor Regularly: Regularly inspect your plants for signs of late blight and take prompt action if you detect any symptoms.",

    "Consider Organic Alternatives: If you prefer organic methods, there are some organic fungicides and treatments available for late blight control. Neem oil and copper-based products with organic certifications are options to explore."
]
        }
cure_link={
    "EarlyBlight":[],
    "LateBlight":[
    ("Folio Gold Syngenta", "https://shop.plantix.net/en/products/pesticides/d2d93b04-a5ff-4ea6-acc8-af7e9a70fd55/folio-gold-syngenta/?utm_source=plantix_web&utm_medium=web_library&utm_campaign=ProductSliderOnDiseasePage"),
    
    ("OHHO Dharmaj Crop Guard", "https://shop.plantix.net/en/products/pesticides/29589de8-d208-4fcb-ad59-b7081c922d09/ohho-dharmaj-crop-guard/?utm_source=plantix_web&utm_medium=web_library&utm_campaign=ProductSliderOnDiseasePage"),
    
    ("Amistar Syngenta", "https://shop.plantix.net/en/products/pesticides/f3251fb6-2a34-4478-bbf7-33a535ed725b/amistar-syngenta/?utm_source=plantix_web&utm_medium=web_library&utm_campaign=ProductSliderOnDiseasePage"),
    
    ("Zampro BASF", "https://shop.plantix.net/en/products/pesticides/fedb41d9-25f0-49b7-b53b-d0e146df29af/zampro-basf/?utm_source=plantix_web&utm_medium=web_library&utm_campaign=ProductSliderOnDiseasePage")
]

}

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
    global class_pred
    global confidence
    global access
    confidence=None
    f=request.files.get('file')
    if f:
        image=np.array(Image.open(BytesIO(f.read())))
        image=cv2.resize(image,(256,256))
        image=np.expand_dims(image,axis=0)
        pred=model.predict(image)
        class_pred=classes[np.argmax(pred)]
        confidence=round(np.max(pred)*100,2)
        access=True
    return render_template("predict.html")
@app.route('/result',methods=["GET","POST"])
def result():
    a=request.form.get("Hi")
    print(a)
    global confidence
    if confidence!=None:
        if(class_pred=="Healthy"):
            healthy="Healthy"
            disease=False
            return render_template("result.html",healthy=healthy,disease=disease)
        else:
            healthy="Not Healthy"
            disease=True
            disease_cure=cure[class_pred]
            disease_cure_links=cure_link[class_pred]
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



if __name__=="__main__":
    app.run(debug=True)
