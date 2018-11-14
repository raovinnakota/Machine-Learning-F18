# Reimplementation of backprop to explore gradient
# checking.
# AUTHOR: Rao Vinnakota

import numpy as np
DEBUG = False


## Various activation functions
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def dlogistic(z):
    #x = logistic(z)
    x = z
    return x * (1.0 - x)

def dtanh(z):
    ## TODO
    #derivative of tanh = 1 - tanh^2(x)
    #tanh is already evaluated
    return 1 - (z**2)

class mlp:
    '''
    A Multi-Layer Perceptron with L inputs, M1 hidden1, M2 hidden2 and
    N output nodes
    '''

    def __init__(self,inputs,targets,
                 nhid1,nhid2,
                 outtype='logistic',
                 hidtype='tanh'):
        '''
        inputs: PxL for P patterns, each of size L
        targets: PxN array for P patterns, each of size L
        nhid1: integer number of hidden units, M1
        nhid2: integer number of hidden units, M2
        outtype: for output units
        '''
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhid1 = nhid1
        self.nhid2 = nhid2

        self.outtype = outtype
        self.hidtype = hidtype
        self.outfn,self.out_deriv = self.select_fn(outtype)
        self.hidfn,self.hid_deriv = self.select_fn(hidtype)

        # Initialise network with random weights
        # Wts format is JxI, where J is the number of receiving units
        # and I is the number of sending units (which includes a bias
        # appended to the layer of units in fwd().
        self.weights1 = (np.random.rand(self.nhid1,self.nin+1)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhid2,self.nhid1+1)-0.5)*2/np.sqrt(self.nhid1)
        self.weights3 = (np.random.rand(self.nout,self.nhid2+1)-0.5)*2/np.sqrt(self.nhid2)

    def select_fn(self,unittype):
        '''return unit func and its derivative'''
        if unittype == 'logistic':
            return logistic,dlogistic
        elif unittype == 'tanh':
            return np.tanh,dtanh
        else:
            print("Type unkown")
            return None,None

    def addbias(self,a):
        '''Append bias term to one-dim array'''
        return np.append(a,1.0)

    def fwd(self,input):
        """
        Forward propagation.
        Iterate one input pattern, updating all network activations.
        returns array of outputs for the inputs.  Stores computations
        for later use in bprop.
        """
        self.input = self.addbias(input)
        self.hid1 = self.hidfn(np.matmul(self.weights1,self.input))
        self.hid1 = self.addbias(self.hid1)
        self.hid2 = self.hidfn( np.matmul(self.weights2,self.hid1) )
        self.hid2 = self.addbias(self.hid2)
        self.output = self.outfn(np.matmul(self.weights3,self.hid2))
        return self.output

    def debugp0(self):
        if not DEBUG: return
        print('w1',self.weights1)
        print('w2',self.weights2)
        print('w3',self.weights3)
        print('in\t',self.input)
        print('h1\t',self.hid1)
        print('h2\t',self.hid2)
        print('out\t',self.output)

    def debugp1(self):
        if not DEBUG: return
        print('drv',self.out_deriv(self.output))
        print('do',self.delta_out)
        print('wts3',self.weights3)
        print('h2',self.hid2)
        print('h1',self.hid1)
        print('dh2',self.delta_h2.shape,self.delta_h2)
        print('input',self.input)
        print('dh1',self.delta_h1)


    def fit(self,inputs,targets,eta,niterations):
        '''
        Train the network for specified iterations.
        Updates after each pattern.
        eta: learning rate
        niterations: number of epochs
        '''
        change = range(self.ndata)
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))

        for iter in range(niterations):
            for i,x in enumerate(inputs):
                self.fwd(x)
                self.error = 0.5*np.sum((self.output-targets[i])**2)
                print("Iter %6d   Pttn %3d   Error %11.9f" % (iter,i,self.error))
                self.bprop(x,targets[i],eta,False)
                self.errCheck(x, targets[i], h=1e-5)
                self.bprop(x,targets[i], eta, True)


    def errCheck(self, input, target, h=1e-5):
        '''
        Check errors.
        Print the relative error for each learned parameter: dE/dw.
        Relative Error described in
        http://cs231n.github.io/neural-networks-3/#gradcheck
        '''
        #initialize lists for numerical error for a given layer
        numerror1 = []
        numerror2 = []
        numerror3 = []

        #iterate through the weights (2d)
        for x in range(self.weights3.shape[0]):
            #create a list for each "row" of the final vector
            rowerror = []
            for y in range(self.weights3.shape[1]):
                #add h
                self.weights3[x,y] += h
                #find new output
                self.fwd(input)
                #save
                hplus = 0.5*np.sum((self.output-target)**2)
                #readjust to now observe L(x-h). Use 2h because coming from x+h
                self.weights3[x,y] -= (2 * h)
                self.fwd(input)
                #record
                hminus = 0.5*np.sum((self.output-target)**2)
                #calculate numerical gradient for specific edge
                numerror = (hplus - hminus)/(2*h)
                #reset weight back to normal
                self.weights3[x,y] += h
                #append to the row of numerical error
                rowerror.append(numerror)
            #add row to overall list (will convert to vector later)
            numerror3.append(rowerror)
        #same process for both weights2 and weights1
        for x in range(self.weights2.shape[0]):
            rowerror = []
            for y in range(self.weights2.shape[1]):
                self.weights2[x,y] += h
                self.fwd(input)
                hplus = 0.5*np.sum((self.output-target)**2)
                self.weights2[x,y] -= (2 * h)
                self.fwd(input)
                hminus = 0.5*np.sum((self.output-target)**2)
                numerror = (hplus - hminus)/(2*h)
                self.weights2[x,y] += h
                rowerror.append(numerror)
            numerror2.append(rowerror)
        for x in range(self.weights1.shape[0]):
            rowerror = []
            for y in range(self.weights1.shape[1]):
                self.weights1[x,y] += h
                self.fwd(input)
                hplus = 0.5*np.sum((self.output-target)**2)
                self.weights1[x,y] -= (2 * h)
                self.fwd(input)
                hminus = 0.5*np.sum((self.output-target)**2)
                numerror = (hplus - hminus)/(2*h)
                self.weights1[x,y] += h
                rowerror.append(numerror)
            numerror1.append(rowerror)
        #convert the lists to 2d arrays
        numerror3 = np.asarray(numerror3)
        numerror2 = np.asarray(numerror2)
        numerror1 = np.asarray(numerror1)
        #use relerr to calculate the relative error/vectorized
        relerr3 = self.relerror(numerror3, self.up3)
        relerr2 = self.relerror(numerror2, self.up2)
        relerr1 = self.relerror(numerror1, self.up1)
        #Print relative error
        print("RelErr w3", relerr3)
        print("RelErr w2", relerr2)
        print("RelErr w1", relerr1)

    def relerror(self, numlayer, alayer):
        '''
        Calculates relative error given both numerical and analytical gradeint
        '''
        relerror = np.abs(numlayer - alayer)/np.maximum(np.abs(numlayer), np.abs(alayer))
        return(relerror)

    def bprop(self,input,target,eta, update):
        '''
        Backprop error from out to inputs, then update weights.
        target: desired outputs
        eta: learning rate.
        '''
        updatew1 = np.zeros(self.weights1.shape)
        updatew2 = np.zeros(self.weights2.shape)
        updatew3 = np.zeros(self.weights3.shape)

        #Calculating Delta
        self.delta_out = (self.output-target)*self.out_deriv(self.output)
        self.delta_h2 = dtanh(self.hid2) * np.dot(np.transpose(self.weights3), np.transpose(self.delta_out))
        self.delta_h2 = self.delta_h2[:-1]
        self.delta_h1 = dtanh(self.hid1) * np.dot(np.transpose(self.weights2), np.transpose(self.delta_h2))
        self.delta_h1 = self.delta_h1[:-1]

        #Adding extra dimensions
        self.delta_out = np.array(self.delta_out)[np.newaxis]
        self.delta_h2 = np.array(self.delta_h2)[np.newaxis]
        self.delta_h1 = np.array(self.delta_h1)[np.newaxis]

        #gradient for each layer
        updatew3 = self.hid2 * np.transpose(self.delta_out)
        updatew2 = self.hid1 * np.transpose(self.delta_h2)
        updatew1 = self.input * np.transpose(self.delta_h1)

        #set the gradient value for error checking
        self.up3 = updatew3
        self.up2 = updatew2
        self.up1 = updatew1

        #update weights if and only if we choose to use backprop
        if update == True:
            self.weights1 -= eta * updatew1
            self.weights2 -= eta * updatew2
            self.weights3 -= eta * updatew3

if __name__ == "__main__":
    np.random.seed(5) # for debugging!
    TEST_TYPE = 'xor' # for debugging!

    anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    # testdata with two outputs
    tdata = np.array([[0,0,0,0],[0,1,0,1],[1,0,0,1],[1,1,1,1]])

    if TEST_TYPE == 'xor':
        q = mlp(inputs=xordata[:,0:2],
                targets=xordata[:,2:3],
                nhid1=2,nhid2=2,
                outtype='logistic',
                hidtype='tanh')
        q.fit(tdata[:,0:2],xordata[:,2:4],0.1,5000)
    elif TEST_TYPE == 'test2':
        q = mlp(inputs=tdata[:,0:2],
                targets=tdata[:,2:4],
                nhid1=2,nhid2=2,
                outtype='logistic',
                hidtype='tanh')
        q.fit(xordata[:,0:2],xordata[:,2:3],0.1,1)
