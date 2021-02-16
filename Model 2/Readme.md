<h1> Model 2 </h1>
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
