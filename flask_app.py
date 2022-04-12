from flask import Flask, render_template, request, redirect, url_for
from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction
from generate_html import generate_html
from model import EncoderDecoder
import json
import torch
from get_loader import dataset, data_loader
app = Flask(__name__)

# mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model
final_model = EncoderDecoder(
    embed_size=300,
    vocab_size=len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

PATH = './models/model.pt'
final_model.load_state_dict(torch.load(PATH, map_location=device))

# use the pre-trained model

final_model.eval()

# define the function to get the images from the url and predicted the class
def get_image_class(dir):
    # get images from the URL and store it in a given path
    get_images(dir)
    # predict the image class of the images with provided directory
    path = get_path(dir)
    images_with_tags = get_prediction(final_model, path)
    # generate html file to render once we predict the classes
    generate_html(images_with_tags)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        # if search button hit, call the function get_image_class
        get_image_class(user)
        #render the image_class.html
        return redirect(url_for('success', name=get_directory(user)))


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8080)
