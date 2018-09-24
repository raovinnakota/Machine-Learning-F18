"""
    Run pruning on mushroom data.
"""

from numpy import *
import dtree


def score(msg,dtree,tree,data,tchr):
    res = dtree.classifyAll(tree,data)
    correct,num = dtree.score_tree(tchr,res)
    print("Num nodes",dtree.countNodes(tree))
    print(msg,'%5d\t%5d\t%6.3f Pct'%(correct,num,100.0*correct/float(num)))
    
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
tree.prune(t,mush,classes)
score('Post-pruning valid correct',tree,t,mush,classes)

mush,classes,features = tree.read_ssvdata('mush_test.ssv')
score('Post-pruning test correct',tree,t,mush,classes)
