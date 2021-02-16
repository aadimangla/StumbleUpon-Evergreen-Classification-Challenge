My First approach was to just take make the predictions based on boilerplate data and ignore rest of the columns as textual data in case is more relevant than any other data. So , I took the text data, split the data into train and validation data. After that I cleaned the data and tokenized it taking vocabulary of 20000 words to remove words which don not occur very often. After that I used Bidirectional LSTM layers in my model along with dense layers. I also used dropout layer to drop some of the units. I used RMSProp optimizer which has momentum, learning rate and rho attributes which we can tune. I used accuracy, AUC, precission and recall as metrics. I recived the following results:
<hr>
Training data : <br>
accuracy: 0.9238 <br>
                AUC: 0.9238 <br>
                precision_2: 0.9303 <br>
                recall_2: 0.9209
<hr>
Validation Data : <br> val_accuracy: 0.7654 <br> val_AUC: 0.7682 <br> val_precision_2: 0.7620 <br> val_recall_2: 0.7813
