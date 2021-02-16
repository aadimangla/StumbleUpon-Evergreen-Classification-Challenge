# StumbleUpon-Evergreen-Classification-Challenge
StumbleUpon Evergreen Classification Kaggle Challenge

<h2> Model 1 </h2>

My First approach was to just take make the predictions based on boilerplate data and ignore rest of the columns as textual data in case is more relevant than any other data. So , I took the text data, split the data into train and validation data. After that I cleaned the data and tokenized it taking vocabulary of 20000 words to remove words which don not occur very often. After that I used <u><b>Bidirectional LSTM layers in my model along with dense layers</b></u>. I also used dropout layer to drop some of the units. I used RMSProp optimizer which has momentum, learning rate and rho attributes which we can tune. I used accuracy, AUC, precission and recall as metrics. I recived the following results:
<hr>
Training data : <br>
- accuracy: 0.9238 <br>
- AUC: 0.9238 <br>
- precision_2: 0.9303 <br>
- recall_2: 0.9209
<hr>
Validation Data : <br> 
- val_accuracy: 0.7654 <br> 
- val_AUC: 0.7682 <br> 
- val_precision_2: 0.7620 <br> 
- val_recall_2: 0.7813

<hr>
<hr>
<h2> Model 2 </h2>
My second approach includes all the variables excluding url and url_id. I used the first model to compute probabilities for boilerplate data and replaced the boilerplate text with those probabilities. I used labelencoder to handle categorical variable. I split the data into training and validation dataset. I used dense layers in my model with <b>tanh and selu activation function<b>. I used RMSProp as optiizer and accuracy , AUC, Precission and recall as metrics. Following are the results:
<hr>
Training data:
<br>
- accuracy: 0.5132<br>
- AUC: 0.5005<br>
- precision: 0.5129 <br>
- recall: 1.0000
<hr>
Validation Data:
<br>
- val_accuracy: 0.5172<br>
- val_AUC: 0.5014<br>
- val_precision: 0.5166<br>
- val_recall: 1.0000
