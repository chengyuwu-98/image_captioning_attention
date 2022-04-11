# Image Captioning including Attention

to be updated later...

## Dataset

[Flickr8k dataset from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## Getting started in the local machine:

1. Clone this repo
2. Download the "Image" folder and captions.txt from the above dataset and put them in a new "flickr8k" folder within the repo directory
3. Download [my trained model](https://drive.google.com/file/d/1t3QbSauxSnZhXE1DbuGwiT2AokOsqOjA/view?usp=sharing) and put it in a new "models" folder within the repo directory
4. pip install -r requirements.txt
5. Open Python Shell and run:
    import spacy
    
    from spacy.cli.download import download
    download(model="en_core_web_sm")
6. Run app.py


## Getting started in Colab:

[![My Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z1sI5wVmoflOggLfIuIIj7qQ0xAICtgn?usp=sharing) 

Download the Image folder and captions.txt from the above dataset, zip the "Image" folder and put it in the Colab working directory with captions.txt
