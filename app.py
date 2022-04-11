import torch
import matplotlib.pyplot as plt
import torch
from model import EncoderDecoder
from get_loader import dataset, data_loader

#testing the app in local env

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the model
    final_model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
    ).to(device)

    
    PATH = './models/model.pt'
    final_model.load_state_dict(torch.load(PATH, map_location=device))

  
    def show_image(img, title=None):
        
        """Imshow for Tensor."""
        
        #unnormalize 
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224 
        img[2] = img[2] * 0.225 
        img[0] += 0.485 
        img[1] += 0.456 
        img[2] += 0.406
        
        img = img.numpy().transpose((1, 2, 0))
        
        
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        
    #generate caption
    def get_caps_from(features_tensors):
        #generate the caption
        final_model.eval()
        with torch.no_grad():
            features = final_model.encoder(features_tensors.to(device))
            caps,alphas = final_model.decoder.generate_caption(features,vocab=dataset.vocab)
            caption = ' '.join(caps)
            show_image(features_tensors[0],title=caption)
    
        return caps,alphas
    
    #show any 1
    dataiter = iter(data_loader)
    images,_ = next(dataiter)

    img = images[0].detach().clone()
    get_caps_from(img.unsqueeze(0))
    
    
if __name__ == '__main__':
    main()
    
 
