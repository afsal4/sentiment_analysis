from django.shortcuts import render
import torch
import torch.nn.functional as F
from .text_preprocessor import Text_preprocessor
from .encoder_pipeline import get_sentiment
import boto3
import torch
from io import BytesIO

# Specify your bucket name and the model file key
bucket_name = 'mlmodels123'
lstm_path = 'sentiment_lstm_cpu.pt'

def download_and_load(model_name, bucket_name, model_file_key):
    # Set up S3 client
    s3 = boto3.client('s3')
    # Get the model file as a byte stream
    buffer = BytesIO()
    s3.download_fileobj(bucket_name, model_file_key, buffer)
    # Load the model into RAM
    buffer.seek(0)  # Go to the start of the BytesIO buffer
    if model_name == 'lstm':
        model = torch.jit.load(buffer)
        return model
    weights = torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
    return weights
   
LSTM_MODEL = download_and_load('lstm', 'mlmodels123', lstm_path)
LSTM_PREPROCESSOR = Text_preprocessor()
LSTM_PREPROCESSOR.max_length = 240


# Create your views here.
def home(request):
    sentiment = None
    prob = None
    pie_out = []
    text = None
    tkn_len = None
    er_flag = 0
    if request.method == 'POST':
        text = request.POST['s-text']
        if text:
            model_type = request.POST['model']

            if model_type == 'Encoder':
                sentiment, prob, tkn_len = get_sentiment(text)
            else:
                try:
                    sentiment, prob, tkn_len = predict(LSTM_MODEL, text, LSTM_PREPROCESSOR)
                except AttributeError:
                    er_flag = 1
            if er_flag != 1:
                pie_out = [
                    { "label": "Positive", "y": round((prob[1]*100)) },
                    { "label": "Negative", "y": round((prob[0]*100)) },
                ]
                text = text.capitalize()
            else:
                text = None
        else: 
            text = None
    context = {'sentiment': sentiment,
               'pie_out': pie_out,
               'tkn_usd': tkn_len, 
               'sentence': text,
               }
    return render(request, 'home/sentiment.html', context)


def predict(model, text, preprocessor):
    labels = ['Negative', 'Positive']
    des_vec, length = preprocessor.description_to_vector(text)
    padded_res = preprocessor.vector_padding(des_vec).squeeze()
    padded_res = padded_res.unsqueeze(0)
    with torch.no_grad():
        res = model(padded_res)
        percentage = F.softmax(res, dim=1)
        forward = torch.argmax(F.softmax(res, dim=1), dim=1)
    return labels[forward], percentage.reshape(-1).tolist(), length
   

