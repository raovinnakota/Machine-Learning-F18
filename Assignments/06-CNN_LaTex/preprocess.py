# Rao Vinnakota
# HW 6
import ast
import time
import cv2
import numpy as np
from im2latex import tokenize_formula, remove_invisible, normalize_formula

'''
This file reads through the various list files and has the code to create a
training, testing and validation set for the model. To tokenize formulas, it
uses code from im2latex, which can be found with the given data set.
'''

IMG_SIZE = 50

def create_list(filename):
    '''Reads files and creates a list, with each element
    containing a line of the file'''
    my_file = open(filename, 'r', encoding = "ISO-8859-1")
    lines = []
    for line in my_file:
        if line[-1] == '\n':
            lines.append(line.rstrip())
    return(lines)

def create_formula(formula):
    '''Creates a formula from tokenizer fxns'''
    formula = remove_invisible(formula)
    formula = normalize_formula(formula)
    tokens = tokenize_formula(formula)
    return(tokens)

def create_set(set_type, formula_list, num, dict):
    '''Creates a training/testing/validation set. Takes
    the list for the set, as well as the list of formulas.
    Returns a list of pairs (image,formula)'''
    formulas = []

    for count in range(num):
        set = set_type[count].split(' ')
        num = ast.literal_eval(set[0])
        image_path = "formula_images/" + set[1] + ".png"
        img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_data = img_data[ 350:550, 200:1500]
        img_data = cv2.resize(img_data,(IMG_SIZE, IMG_SIZE))
        formula = create_formula(formula_list[num])
        if (len(formula) < 50):
            formula = formula + (['\\'] * (50-len(formula)))
        else:
            formula = formula[:50]
        formlist = change_values(formula, dict)
        formlist = np.transpose(formlist)
        formulas.append([np.array(img_data), formlist.flatten('F')])
    return(formulas)

def create_token_dict(formula_list):
    '''Creates two dictionaries for LaTeX tokens.
    Each token is assigned a unique integer. The neural network
    can process integers
    '''
    int_to_form = {}
    form_to_int = {}
    num = 0
    for formula in formula_list:
        temp = create_formula(formula)
        for i in temp:
            if i in form_to_int:
                pass
            else:
                int_to_form[num] = i
                form_to_int[i] = num
                num += 1
    return(form_to_int, int_to_form)

def int_to_dict(list, dict):
    '''Changes integer value to LaTeX tokens'''
    out = []
    for i in list:
        if i in dict:
            out.append(dict[i])
        else:
            out.append(1)
    return(out)

def change_values(list, dict):
    '''Changes LaTeX tokens to integer array'''
    out = []
    for i in list:
        inner_out = [0] * len(dict)
        if i in dict:
            num = dict[i]
            inner_out[num] = 1
        out.append(inner_out)
    return(np.array(out))
