# Rao Vinnakota
# HW 6

import numpy as np
import os, sys
import tensorflow as tf
import tflearn
import Levenshtein
import argparse
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from preprocess import create_list, create_set, create_token_dict, int_to_dict

#Global variable for model
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = "image2latex"
EPOCHS = 10
SAVE = True
MODEL_PATH = ''

######Parsing Arguments######################
#creating parser for sys input
parser = argparse.ArgumentParser(description='Build and/or test a DNN')
#adding argument for epochs, files, and model
parser.add_argument('-n', help='Number of epochs')
parser.add_argument('-m', help='Trained Model')
parser.add_argument('test_image_list', nargs='?', help="Test image list")
parser.add_argument('formula_list', nargs='?', help="List of formulas")
args = vars(parser.parse_args())
print(args)
if (args['n']):
    EPOCHS = int(args['n'])
if (args['m']):
    SAVE = False
    MODEL_PATH = args['m']

######Creating/Importing Testing Data########
train_list = create_list('im2latex_train.lst')
test_list = create_list(args['test_image_list'])
validate_list = create_list('im2latex_validate.lst')
formula_list = create_list('im2latex_formulas.lst')
test_formula_list = create_list(args['formula_list'])
form_to_int, int_to_form = create_token_dict(formula_list[:400])

formula_train = create_set(train_list, formula_list, 500, form_to_int)
formula_test = create_set(test_list, test_formula_list, 200, form_to_int)
formula_validate = create_set(validate_list, formula_list, 200, form_to_int)



######Creating the Training and Validation Set for TfLearn###################
X_train = np.array([i[0] for i in formula_train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_train = np.array([i[1] for i in formula_train])

X_valid = np.array([i[0] for i in formula_validate]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_valid = np.array([i[1] for i in formula_validate])


#######Building a CNN#######################
#Train only if a model isn't loaded

#reset to overwrite any lingering data
tf.reset_default_graph()

#adding input in shape 50 x 50
convnet = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1], name='input')
#convolution and pool layers
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
#output layer, expect 277 * 50 output factors
convnet = fully_connected(convnet, 13850, activation = 'softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate=LR,
                    loss='binary_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

if (SAVE==False):
    print("Sorry Sven, I'm having a real hard time loading a model! So here's a quick 2 epoch train")
    EPOCHS=2
    #model = model.load(MODEL_PATH)

#fiting model, chose binary_crossentropy as loss fxn
model.fit({'input': X_train}, {'targets':Y_train}, n_epoch=EPOCHS,
                validation_set=({'input': X_valid}, {'targets':Y_valid}),
                snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
if (SAVE): #if you want to save new trained model
        model.save('model.tflearn')

#######Testing Neural Net################
#different comparison points
token_matches = 0
exact_matches = 0
tot_distance = 0
lev_dist = 0

#Iterate throught he testing set
for test in formula_test:
    data = test[0].reshape(IMG_SIZE, IMG_SIZE, 1)
    #make prediction
    prediction = model.predict([data])
    #reshape to 2d numpy array to widthdraw information
    prediction = prediction.reshape(50,277)
    actual = test[1].reshape(50,277)
    tokens = []
    actual_tok = []
    #choose symbol that had the highest probability predicted
    for i in prediction:
        tokens.append(np.argmax(i))
    for i in actual:
        actual_tok.append(np.argmax(i))
    #use dictionary to create tokens integers predicted
    tokens = int_to_dict(tokens, int_to_form)
    actual_token = int_to_dict(actual_tok, int_to_form)
    #compare the two token lists
    for i in range(50):
        #seeing how mnay individual tokens match exactly
        if tokens[i] == actual_token[i]:
            tot_distance += 1
    #checks if full expression of tokens match
    if tokens == actual_token:
        token_matches += 1
    #join tokens to evaluate string
    expression = ''.join(tokens)
    actual_exp = ''.join(actual_token)
    #See if actual string matches
    if (expression == actual_exp):
        exact_matches += 1
    #Check levenshtein distance between the predicted and actual formula
    lev_dist += Levenshtein.distance(expression, actual_exp)


token_matches = float(token_matches)/float(len(formula_test))
exact_matches = float(exact_matches)/float(len(formula_test))
ave_distance = float(tot_distance)/50.0
ave_lev_dist = float(lev_dist)/50.0

#print output
print("(Exact match): " + str(exact_matches) + "% (Token): " + str(token_matches) + "%")
print("(Average Individual Token Matches): " + str(ave_distance) +
        " (Average Levenshtein Distance): " + str(ave_lev_dist))
