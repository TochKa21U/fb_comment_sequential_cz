# Introduction
Comment classifier AI Model
## Object

Automatically classify comments into sentiments (Positive, Neutral, Negative) to help the product team quickly gauge public reception.

# Solution Overview

AI model is a automated systen that trained on previous supplied data to make decision on newly given data, aka predicting a result from previous learnings.

**We have used the provided base
model since it has passed all internal security assessments on biases and discrimination.**

Therefore using our internal model(Electra Architecture 13.8 Million Parameter), we have fine tune it to make decision about given comment whether it is :

- "POSITIVE"
- "NEGATIVE"
- "NEUTRAL".

This decision will help our customers to identify product feedback from their own customers and see how satisfied are the customers. It can be also used with customer churn rate

## Development Process

Model has been developed by using base model, Huggingface Library Transformer model and Huggingface datasets library. 

Hugginface provided libraries are all open source and can be used in business without any licensing problem.

`Note : Only proprietary software would be models which is provided by individuals or company may be subjected to copyright. However our internal model would be safe to use. `

After model is trained with the data, we have packaged it to be able to easly consumable with any library. 

For recommended methods please take a look at the `README.md` file.

## Model Metrics

We have achieved around 80% accuracy with this model, which means that out of 10 prediction, 8 predictions should be correct.

If business requirements demands higher precision/accuracy of the model, model can be trained different parameters, different datasets, and even with the different base model with higher number of parameters.

Generally speaking higher parameter models tends to perform better.

## Business Impact and Next Steps

This model can help business with analyzing customer feedback, and resolve the customer issues much more quickly by forwarding customer support agent/social media manager attention to negative comments to help the resolve issue of the customer.

This can be also used as marking total number of negative comments , neutral and positive comments to get a feedback about the item directly from the market.

Further improvement can be done by training this model with more data from customers as well