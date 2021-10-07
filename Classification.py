import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('credit_card_approval_classification.csv')

# dataset.isna().sum()

dataset = dataset.dropna(axis=0, inplace=False)

input_data = dataset[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_PHONE', 'JOB', 'STATUS']]
output_data = dataset[['TARGET']]

# Normalize Data
conv = OrdinalEncoder()
input_data = conv.fit_transform(input_data)
# print(input_data)

scal = MinMaxScaler()
input_data = scal.fit_transform(input_data)
# print(input_data)


ohe = OneHotEncoder(sparse = False)
output_data = ohe.fit_transform(output_data)
# print(output_data)

pca = PCA(n_components = 3)
input_data = pca.fit_transform(input_data)

layer = {
    'input': 3,
    'hidden': 12,
    'output': 2
}

input_hidden = {
    'weight': tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    'bias': tf.Variable(tf.random_normal([layer['hidden']]))
}

hidden_output = {
    'weight': tf.Variable(tf.random_normal([layer['hidden'], layer['output']])),
    'bias': tf.Variable(tf.random_normal([layer['output']]))
}

def activation_function(output):
    return tf.nn.sigmoid(output)

def feed_forward(input_data):
    data_x1 = tf.matmul(input_data, input_hidden['weight'] + input_hidden['bias'])
    data_y1 = activation_function(data_x1)

    data_x2 = tf.matmul(data_y1, hidden_output['weight'] + hidden_output['bias'])
    data_y2 = activation_function(data_x2)

    return data_y2

input_place = tf.placeholder(tf.float32, [None, layer['input']])
output_place = tf.placeholder(tf.float32, [None, layer['output']])

y_pred = feed_forward(input_place)
error = tf.reduce_mean((output_place - y_pred) ** 2)

learning_rate = 0.05
training = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=2/7, random_state=1)

epoch = 2000
save_model = tf.train.Saver()

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, epoch+1):
        train_dict = {
            input_place: x_train,
            output_place: y_train
        }

        sess.run(training, feed_dict=train_dict)
        loss = sess.run(error, feed_dict=train_dict)
        
        if i % 100 == 0:
            print(f'Epoch: {i}, Loss: {loss}')

            if i == 500:
                val_error = loss
                save_model.save(sess, 'model/model.ckpt')
            
            if i % 500 == 0:
                validation_error = loss
                if loss < validation_error:
                  save_model.save(sess, 'model/model.ckpt')

    accuracy = tf.equal(tf.argmax(output_place, axis=1), tf.argmax(y_pred, axis=1))
    result = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    test_dict = {
        input_place: x_test,
        output_place: y_test
    }

    print(f'Accuracy: {sess.run(result, feed_dict=test_dict) * 100}%\n')