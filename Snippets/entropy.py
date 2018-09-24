from math import *

def entropy_total(p_pos, p_neg):
    total = p_pos + p_neg
    prop_pos = float(p_pos)/float(total)
    prop_neg = float(p_neg)/float(total)
    if (prop_pos != 0 and prop_neg != 0):
        entropy = -1 * ((prop_neg * log(prop_neg,2)) + (prop_pos * log(prop_pos, 2)))
    elif (prop_pos != 0 and prop_neg == 0):
        entropy = -1 * ((prop_pos * log(prop_pos, 2)))
    else:
        entropy = -1 *((prop_neg * log(prop_neg, 2)))
    return (entropy)

entropyA1 = (float(4)/float(6)) * entropy_total(3,1) + float(2)/float(6) * entropy_total(1,1)
entropyA2 = (float(4)/float(6)) * entropy_total(2,2) + float(2)/float(6) * entropy_total(2,0)
entropyA3 = (float(3)/float(6)) * entropy_total(3,0) + float(3)/float(6) * entropy_total(1,2)

entropyA3_no = entropy_total(1,2)

print(entropyA3_no)
