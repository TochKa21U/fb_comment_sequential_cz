# Electra Sequence Classifier for Czech Facebook Comments

## Model Description
This model is an ELECTRA-based transformer model fine-tuned for sentiment classification on Facebook comments written in Czech. It categorizes comments into three sentiments: Positive, Neutral, and Negative.

### Model Architecture
The model uses the `ElectraForSequenceClassification` architecture from Hugging Face's Transformers library, fine-tuned on a labeled dataset of Czech Facebook comments.

## How to Use
Below are instructions on how to use this model in Python using the Hugging Face Transformers library.

### Requirements
- Python 3.6+
- Transformers 4.0.0+
- Torch 1.7.0+

### Setup
Install the required libraries using pip:
```bash
pip install transformers torch
```
### Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Use cuda if avaliable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained('tochkamg/electra-sequence-classifier-czech')
model = AutoModelForSequenceClassification.from_pretrained('tochkamg/electra-sequence-classifier-czech')

# Example test
text = "perfektní prace"

# Testing function
def test_loaded_model(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return model.config.id2label[predicted_class_id]

model.to(device)

# Print predicted class labels
test_loaded_model(text)

```
## Prediction and File Structure
- Classification_FB_Comments_Sequential.ipynb : Step by step loading the pre-trained model from model folder, and training it
- Test_Model_Predict.ipynb : Load the model and use it for prediction, purposely handled in seperate file for checking model loading and other stuff
- CSVEdit.ipynb : Data processing for taking labeled_comments.txt and converting into dataframe with labels, purposely put seperately
- Model folders
    - model : Original untrained model, contains pre-trained model and tokenizer info. We always get the tokenizer from here
    - lowest_loss : Model that used in the prediction, which has lower loss value compared to other models
    - sixteen_epoch : Experimental model, tested with different traning arguments
    - outputs : Just a necessary folder for evaluation, contains nothing