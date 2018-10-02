from numpy import *
import dtree
import sys

tree = dtree.dtree()
# data, class of each data item, attribute names
tennis,classes,features = tree.read_data('tennis.data')
t=tree.make_tree(tennis,classes,features)
print('Tree derived from ID3 learning on the data set.')
#print(t)

res = tree.classifyAll(t,tennis)
print( 'Correct %d %d' % tree.score_tree(res,classes))
tree.zeroCountsField(t)
tree.updateCountsAll(t,tennis,classes,majorityupdate=True)
tree.printTree(t,' ')
print("Num nodes",tree.countNodes(t))
'''print('Classification of all examples')
for i in range(len(tennis)):
    tree.classify(t,tennis[i])
'''
tennis,classes,features = tree.read_data('tennprune5.dat')
res = tree.classifyAll(t,tennis)

print('Should make errors..............................')
print( 'Correct %d %d' % tree.score_tree(res,classes))

#tree.addCountsField(t, list(set(classes)))
tree.zeroCountsField(t) #, list(set(classes)))
tree.updateCountsAll(t,tennis,classes,majorityupdate=False)
tree.printTree(t,' ')
#print(t)

tree.numpruned = 0
print("PRUNE=============================")
print(t)
#print('num pruned',tree.numpruned)

tree.printTree(t,' ')
print("Num nodes",tree.countNodes(t))
