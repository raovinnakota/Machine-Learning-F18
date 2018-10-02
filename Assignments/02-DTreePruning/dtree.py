#
# This version will attempt to prune nodes using reduced error pruning.
# Nodes are successively pruned if a validation set gets improved error
# after topmost error-causing node is pruned.
#
# Original code from Chapter 6 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Modified by S. Anderson, 2013, 2018

# Nodes in the tree consist of
#  Attribute
#  Leaf: if a leaf or not
#  Count: dictionary of count examples beneath node
#  Majorclass: Classification of most common example beneath node.
#  Value: Feature value of attribute (only for leaf nodes)
#  OTHER: these are values of the Attributes (aka features) and are all
#         other strings

from numpy import *


def keymax(d):
    '''
    :param d: a dictionary
    :return: dictionary key of maximum value in dictionary d
    '''
    vals=list(d.values())
    keys=list(d.keys())
    return keys[vals.index(max(vals))]

class dtree:
    ''' A basic Decision Tree with ID3 learning.'''
    ATTR = '@ATTR' # constant to indicate attribute for this node.
    LEAF = '@LEAF' # constant to indicate a leaf key of node
    VALUE = '@VALUE' # constant to indicate value key of leaf node
    COUNT = '@COUNT' # constant to indicate count key of leaf node
    MAJORCLASS = '@MAJOR' # constant to indicate dominant class
    NON_ATTRIBUTES = [ATTR,LEAF,VALUE,COUNT,MAJORCLASS]


    def __init__(self):
        """ Constructor """
        self.numpruned = 0
        self.tree = {} # initial representation of decision tree
        self.data = [ ] # data related to tree
        self.classes = [ ] # specified classes for self.data

    def read_ssvdata(self,filename):
        '''Reads data from filename, an ssv file.
        File format:
        1st line number of items per datapoint and another integer that is ignored.
        2nd line is space-delimited hames of class, then attributes
        3rd line is type of each attribute and is ignored.
        4th and later lines are data points: first value is class, followed by attributes
        Returns lists:
        data, class values, attribute names
        '''
        fid = open(filename,"r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(" "))
        fid.close()

        self.AttrNames = data[1]
        self.AttrNames = self.AttrNames[1:] # drop first column
        data = data[3:]  # now we want data only, drop first 3 lines
        self.classes = []
        # get cols 1-22 into data, col 0 is the class
        for d in range(len(data)):
            self.classes.append(data[d][0])
            data[d] = data[d][1:]
        self.data = data
        return data,self.classes,self.AttrNames

    def read_data(self,filename):
        '''Reads data from filename.
        File format:
        1st line is comma-delimited list of attribute names, last column is class
        Later lines are attribute values, last column is class value
        Returns lists:
        data, class values, attribute names
        '''
        fid = open(filename,"r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))
        fid.close()

        self.AttrNames = data[0]
        self.AttrNames = self.AttrNames[:-1] # drop final column
        data = data[1:]  # now it's data only, no attribute names
        self.classes = []
        for d in range(len(data)):
            self.classes.append(data[d][-1])
            data[d] = data[d][:-1]
        self.data = data
        return data,self.classes,self.AttrNames

    def isLeaf(self,tree):
        '''Returns true iff tree nodes is a leaf.
        :param tree: tree must be a dictionary.
        '''
        return tree.get(self.LEAF,False)

    def leafValue(self,tree):
        '''
        :return: value of leaf or majority class
        :param tree: tree must be a dictionary.
        '''
        return tree.get(self.VALUE,self.MAJORCLASS)

    def countNodes(self,tree):
        ''':return: count of all internal nodes and leaves in tree.'''
        if self.isLeaf(tree): return 1
        sum = 1 # count this node
        for f in self.getFeatures(tree):
            sum += self.countNodes(tree[f])
        return sum


    def classify(self,tree,datapoint):
        '''
        :param tree: a tree, stored as a dictionary (or string for leaves)
        :param datapoint: an ordered list of attribute values.
        '''

        if self.isLeaf(tree): # Have reached a leaf
            return self.leafValue(tree) # return value of leaf
        else:
            attr = self.getAttribute(tree)
            i = self.AttrNames.index(attr)
            #print('following',datapoint[i])
            t = tree.get(datapoint[i],None) # follow this attr-value of datapoint
            if (t == None): # no match, return majority class
                return tree[self.MAJORCLASS]
            else:
                return self.classify(t,datapoint)

    def classifyAll(self,tree,data):
        '''Classify all examples in the list, data.
        :param tree: current tree
        :param data: data to classify
        '''
        return [ self.classify(tree,data[i]) for i in range(len(data))]

    def getAttribute(self,tree):
        '''
        :return: attribute name of tree's root node.
        '''
        return tree[self.ATTR]

    def isFeature(self,f):
        '''
        :return: true if f is not in non_attributes list,
        and is not deleted.
        :param f: feature value to test
        '''
        return (f not in self.NON_ATTRIBUTES)

    def getFeatures(self,tree):
        '''Return list of all attribute keys for tree.'''
        return [k for k in tree.keys() if self.isFeature(k)]

    def computeTotalEntropy(self,newClasses,classes,nData):
        '''
        Returns class with most entropy and value of that entropy.
        :param newClasses: is set of all classes
        :param classes: is actual class
        membership data,
        :param nData: is num of examples.
        '''
        index = 0
        totalEntropy = 0
        # Compute the default class (and total entropy)
        frequency = zeros(len(newClasses))

        for aclass in newClasses:
            frequency[index] = classes.count(aclass)
            totalEntropy += self.calc_entropy(float(frequency[index])/nData)
            index += 1
        default = classes[argmax(frequency)]

        return default,totalEntropy

    def newLeaf(self,value,count):
        '''Create a new leaf with specified value and count.'''
        leaf = {}
        leaf[self.VALUE] = value
        leaf[self.LEAF] = True
        leaf[self.COUNT] = {}
        leaf[self.COUNT][value] = count
        leaf[self.MAJORCLASS] = None
        return leaf

    def newNode(self,attrvalue,count):
        '''Create a new node with specified attribute value and count.'''
        node = {}
        node[self.ATTR] = attrvalue
        node[self.LEAF] = False
        node[self.COUNT] = {}
        node[self.MAJORCLASS] = None
        return node

    def make_tree(self,data,classes,AttrNames,maxlevel=-1,level=0):
        """
        The main function to construct the tree.
        :param data:  is list of lists, each sublist is one example.
        :param classes: are values, one for each example.
        :param AttrNames: is list of all names.
        """
        nData = len(data)
        nAttributes = len(data[0])

        try:
            self.AttrNames
        except:
            self.AttrNames = AttrNames

        # Create lst of classes; maintains order!
        newClasses = [ ]
        for aclass in classes:
            if aclass not in newClasses:
                newClasses.append(aclass)

        default,totalEntropy = self.computeTotalEntropy(newClasses,classes,nData)

        if nData==0 or nAttributes == 0 or (maxlevel>=0 and level>maxlevel):
            # Have reached an empty branch, return best class as [class 0]
            return self.newLeaf(value=default,count=1)
        elif classes.count(classes[0]) == nData:
            # Only 1 class remains
            return self.newLeaf(value=default,count=1)
        else:
            # Choose which feature is best
            gain = zeros(nAttributes)
            for feature in range(nAttributes):
                g = self.calc_info_gain(data,classes,feature)
                gain[feature] = totalEntropy - g

            bestAttr = argmax(gain)
            #init the tree as a single node
            #print('Best attribute',AttrNames[bestAttr])
            tree = self.newNode(attrvalue=AttrNames[bestAttr],count=1)

            # List the values that bestAttr can take.
            # This decreases as we descend the tree.
            values = []
            for datapoint in data:
                if datapoint[bestAttr] not in values:
                    values.append(datapoint[bestAttr])

            for value in values:
                # Find the datapoints with each attr value
                newData = []
                newClasses = []
                index = 0

                for datapoint in data:
                    if datapoint[bestAttr]==value:
                        # extract bestAttr value, leave all others!
                        if bestAttr==0:
                            newdatapoint = datapoint[1:]
                            newNames = AttrNames[1:]
                        elif bestAttr==nAttributes:
                            newdatapoint = datapoint[:-1]
                            newNames = AttrNames[:-1]
                        else:
                            newdatapoint = datapoint[:bestAttr]
                            newdatapoint.extend(datapoint[bestAttr+1:])
                            newNames = AttrNames[:bestAttr]
                            newNames.extend(AttrNames[bestAttr+1:])
                        newData.append(newdatapoint)
                        newClasses.append(classes[index])

                    index += 1 # index of the datapoint

                # Now recurse to the next level
                subtree = self.make_tree(newData,newClasses,newNames,maxlevel,level+1)
                tree[value] = subtree
            return tree

    def printTree(self,tree,str):
        '''Print tree.
        '''
        if self.isLeaf(tree):
            print (str, "\t->\t", tree[self.VALUE], tree[self.MAJORCLASS], tree[self.COUNT])
        else: # not leaf
            print(str,tree[self.ATTR],tree[self.MAJORCLASS],tree[self.COUNT])
            for a in self.getFeatures(tree):
                print (str, a)
                self.printTree(tree[a], str + "\t")

    def calc_entropy(self,p):
        if p!=0:
            return -p * log2(p)
        else:
            return 0

    def calc_info_gain(self,data,classes,feature):
        '''Calculates the information gain based on entropy.
        data: list of examples (each example is a list of feature values
        classes: list of class names
        feature: feature index (an int) for which gain is calculated
        '''
        gain = 0
        nData = len(data)

        # values is list of values that feature can take
        values = []
        # this loop adds to values any features not already in values
        for datapoint in data:
            if values.count(datapoint[feature])==0:
                values.append(datapoint[feature])

        featureCounts = zeros(len(values)) # zeros makes a list/vector of zeros
        entropy = zeros(len(values))
        valueIndex = 0
        # Find where those values appear in data[feature] and the corresponding class
        for value in values:
            dataIndex = 0
            newClasses = []
            # Explain the purpose of this loop.
            for datapoint in data:
                if datapoint[feature]==value:
                    featureCounts[valueIndex]+=1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1

            # Get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass)==0: # why add them if count is zero?
                    classValues.append(aclass)

            classCounts = zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex]+=1
                classIndex += 1

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex])/sum(classCounts))

            # Computes the entropy
            gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            valueIndex += 1
        return gain

    def score_tree(self,x,y):
        '''
        :param x: first list of values
        :param y: second list of values
        :returns: (number correct,number tested)
        '''
        numcorrect = 0
        n = len(x)
        for i in range(n):
            if x[i] == y[i]:
                numcorrect += 1
        return (numcorrect,n)

    def zeroCountsField(self,tree):
        '''Zero the counts field of each node.'''

        if self.isLeaf(tree):
            for k in tree[self.COUNT].keys():
                tree[self.COUNT] = {}
        else:
            for k in tree[self.COUNT].keys():
                tree[self.COUNT][k] = 0
            for item in tree.keys():
                if self.isFeature(item):
                    self.zeroCountsField(tree[item])

    def updateMajorityClass(self,tree):
        '''
        Set the majority to be the class with the most
        examples under this node.
        '''
        tree[self.MAJORCLASS] = keymax(tree[self.COUNT])

    def updateCounts(self,tree,datapoint,classpoint,majorityupdate):
        '''
        Adds count of
        class point to all nodes leading to its classification in a leaf.
        :param tree: a tree, stored as a dictionary (or string for leaves)
        :param datapoint: an ordered list of attribute values.
        :param classpoint: class membership of datapoint
        :param majorityupdate: Also update majority field.
        '''

        tree[self.COUNT][classpoint] = tree[self.COUNT].get(classpoint,0) + 1
        if majorityupdate:
            self.updateMajorityClass(tree)

        if self.isLeaf(tree):
            return
        else:
            attr = self.getAttribute(tree)
            for i,aname in enumerate(self.AttrNames):
                if aname == attr: break
            # i is now value for attribute attr
            try:
                t = tree[datapoint[i]] # follow this attr-value of datapoint
                self.updateCounts(t,datapoint,classpoint,majorityupdate)
            except:
                return # failed to update

    def updateCountsAll(self,tree,data,classes,majorityupdate):
        '''Update tree node counts with all data.'''
        for i in range(len(data)):
            self.updateCounts(tree,data[i],classes[i],majorityupdate)

## Add pruning code here.

    def prune(self,tree,data,classes):
        '''Prunes tree in which nodes have counts for all classes over a
        validation set.  For each node consider number of errors
        caused by using majority class to label all examples beneath
        that node.  Selects the pruning the yields the greatest
        increase in num correct.
        :param tree: Initial unpruned tree
        :param data: All input examples to be used during pruning.
        :param classes: Assigned classes from the input data.
        '''
        # TODO
        tree['@LEAF'] = "Yes"
        tree['@VALUE'] = tree['@MAJOR']
        for i in self.getFeatures(tree):
            del(tree[i])
        if ('@ATTR' in tree.keys()):
            del(tree['@ATTR'])
        return(tree)



    def countErrors(self):
        '''
        :return : Errors in entire tree using all data.
        '''
        res = self.classifyAll(self.tree,self.data)
        ncorrect,numresults = self.score_tree(res,self.classes)
        return numresults - ncorrect
