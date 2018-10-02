"""
    Run pruning on mushroom data.
"""

from numpy import *
import dtree

def return_score(dtree,tree,data,tchr):
    '''
    returns the number correct from validation set
    identical to score, except with no print
    '''
    res = dtree.classifyAll(tree,data)
    correct,num = dtree.score_tree(tchr,res)
    return (correct/float(num))

def score(msg,dtree,tree,data,tchr):
    res = dtree.classifyAll(tree,data)
    correct,num = dtree.score_tree(tchr,res)
    print("Num nodes",dtree.countNodes(tree))
    print(msg,'%5d\t%5d\t%6.3f Pct'%(correct,num,100.0*correct/float(num)))

def prune_mush(dtree, tree, data, tchr):
    '''Recursively moves through the tree. Tests pruning the node and prunes if
    :param tree: Initial unpruned tree
    :param data: All input examples to be used during pruning.
    :param classes: Assigned classes from the input data.
    '''
    newtree = dict(tree) #creates the new tree using dict() creates a new object
    dtree.prune(newtree, data, tchr) #prune the new tree
    if return_score(dtree,newtree,data,tchr) > return_score(dtree,tree,data,tchr):
        print("Prune: " + tree['@ATTR']) #compare the scores of the two pruned trees
        dtree.prune(tree, data, tchr) #prune the node if the score of the new tree is higher
        prune_mush(dtree, tree, data, tchr)
    if dtree.getFeatures(tree):
        for i in dtree.getFeatures(tree):
            prune_mush(dtree, tree[i], data, tchr)
    return(tree)

#######################################
# Create tree from training data
#######################################
tree = dtree.dtree()
mush,classes,features = tree.read_ssvdata('mush_train.ssv')
t=tree.make_tree(mush,classes,features)
#print('Tree derived from ID3 learning on the data set.')
#tree.printTree(t,' ')

# Counts field used for majority voting.  Must be set.
tree.zeroCountsField(t)
tree.updateCountsAll(t,mush,classes,majorityupdate=True)

score('Pre-prune train correct',tree,t,mush,classes)

# Test tree on testing data
mush,classes,features = tree.read_ssvdata('mush_test.ssv')
score('Pre-pruning test correct',tree,t,mush,classes)

#######################################
# Load validation data and prune tree.
#######################################
print('\nPruning===============================')

mush,classes,features = tree.read_ssvdata('mush_valid.ssv')
score('Validation correct',tree,t,mush,classes)

tree.zeroCountsField(t)
tree.updateCountsAll(t,mush,classes,majorityupdate=False)
#newtree = dict(t)
#tree.prune(newtree,mush,classes)
t = prune_mush(tree,t,mush,classes)
#tree.printTree(t, ' ')
score('Post-pruning valid correct',tree,t,mush,classes)

mush,classes,features = tree.read_ssvdata('mush_test.ssv')
score('Post-pruning test correct',tree,t,mush,classes)
