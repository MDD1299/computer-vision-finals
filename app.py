from flask import Flask, render_template
from numpy import mean
from numpy import expand_dims
import base64
import random
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import matplotlib
import numpy as np
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generatedimage/', methods=['GET', 'POST'])
def generatedimage():

    load_latent_points = np.loadtxt('./static/latent_points.txt')
    latent_points = load_latent_points

    def resulting_vec(points, ix):
        a_point = [i-1 for i in ix]
        b_point = points[a_point]
        result_vec = mean(b_point, axis=0)
        return result_vec

    model = load_model('image-generator-model.h5')

    feature_1 = random.sample(range(len(latent_points)), 4)
    feature_2 = random.sample(range(len(latent_points)), 6)
    feature_3 = random.sample(range(len(latent_points)), 9)

    feature1 = resulting_vec(latent_points, feature_1)
    feature2 = resulting_vec(latent_points, feature_2)
    feature3 = resulting_vec(latent_points, feature_3)

    vector = feature1 + feature2 + feature3
    vector = expand_dims(vector, 0)
    gan_image = model.predict(vector)
    gan_image = (gan_image + 1) / 2.0
    plt.imshow(gan_image[0])

    plot_img = 'static/generatedimage.png'
    if os.path.exists(plot_img):
        os.remove(plot_img)
    plt.savefig(plot_img, format='png')

    with open(plot_img, 'rb') as img_file:
        convert_img = base64.b64encode(img_file.read()).decode('utf-8')
    
    return render_template('generate.html', generatedimage=convert_img)

if (__name__ == '__main__'):
    app.run()