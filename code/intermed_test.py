import sys
import time
import numpy as np

from utils import *
from rnnmath import *
from sys import stdout


class RNN(object):
    '''
    This class implements Recurrent Neural Networks.
    
    You should implement code in the following functions:
        predict             ->  predict an output sequence for a given input sequence
        acc_deltas          ->  accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt     ->  accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        compute_loss        ->  compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
        compute_mean_loss   ->  compute the average loss over all sequences in a corpus
        generate_sequence   ->  use the RNN to generate a new (unseen) sequnce
    
    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.
        
        DO NOT CHANGE THIS
        
        vocab_size      size of vocabulary that is being used
        hidden_dims     number of hidden units
        out_vocab_size  size of the output vocabulary
        '''
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.out_vocab_size = out_vocab_size 
        
        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
        self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
        self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)
        
        # matrices to accumulate weight updates
        self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
        self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
        self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

    def apply_deltas(self, learning_rate):
        '''
        update the RNN's weight matrices with corrections accumulated over some training instances
        
        DO NOT CHANGE THIS
        
        learning_rate   scaling factor for update weights
        '''
        # apply updates to U, V, W
        self.U += learning_rate*self.deltaU
        self.W += learning_rate*self.deltaW
        self.V += learning_rate*self.deltaV
        
        # reset matrices
        self.deltaU.fill(0.0)
        self.deltaV.fill(0.0)
        self.deltaW.fill(0.0)
    
    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x
        
        x   list of words, as indices, e.g.: [0, 4, 2]
        
        returns y,s
        y   matrix of probability vectors for each input word
        s   matrix of hidden layers for each input word
        
        '''
        
        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))
        
        for t in range(len(x)):
            ##########################
            # --- your code here --- #
            ##########################
            x_t = make_onehot(x[t], self.vocab_size)
            net_in_t = np.dot(self.V, x_t) + np.dot(self.U, s[t-1])
            s[t] = sigmoid(net_in_t)
            net_out_t = np.dot(self.W, s[t])
            y[t] = softmax(net_out_t)
        
        return y, s

    def compute_loss(self, x, d):
        '''
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x       list of words, as indices, e.g.: [0, 4, 2]
        d       list of words, as indices, e.g.: [4, 2, 3]

        return loss     the combined loss for all words
        '''

        loss = 0.

        ##########################
        # --- your code here --- #
        ##########################
        y, _ = self.predict(x)
        for t in range(len(x)): 
            d_t = make_onehot(d[t], self.out_vocab_size)
            loss -= np.dot(d_t, np.log(y[t]))

        return loss

    def compute_mean_loss(self, X, D):
        '''
        compute the mean loss between predictions for corpus X and desired outputs in corpus D.
        
        X       corpus of sentences x1, x2, x3, [...], each a list of words as indices.
        D       corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.
        
        return mean_loss        average loss over all words in D
        '''
        
        mean_loss = 0.
        n_words = 0
        ##########################
        # --- your code here --- #
        ##########################
        for sentence in range(len(X)):
            mean_loss += self.compute_loss(X[sentence], D[sentence])
            n_words += len(X[sentence])
        
        return mean_loss / n_words

    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x   list of words, as indices, e.g.: [0, 4, 2]
        d   list of words, as indices, e.g.: [4, 2, 3]
        y   predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s   predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''
        
        for t in reversed(range(len(x))):
            ##########################
            # --- your code here --- #
            ##########################     
            d_t = make_onehot(d[t], self.out_vocab_size)
            delta_out = (d_t - y[t]) * np.ones(self.out_vocab_size)
            self.deltaW += np.outer(delta_out, s[t])
            f_net = s[t] * (np.ones(self.hidden_dims) - s[t])
            delta_in = np.dot(self.W.T, delta_out) * f_net
            x_t = make_onehot(x[t], self.vocab_size)
            self.deltaV += np.outer(delta_in, x_t)
            self.deltaU += np.outer(delta_in, s[t-1])

    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x       list of words, as indices, e.g.: [0, 4, 2]
        d       list of words, as indices, e.g.: [4, 2, 3]
        y       predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s       predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps   number of time steps to go back in BPTT
        
        no return values
        '''
        for t in reversed(range(len(x))):
            ##########################
            # --- your code here --- #
            ##########################     
            d_t = make_onehot(d[t], self.out_vocab_size)
            delta_out = (d_t - y[t]) * np.ones(self.out_vocab_size)
            self.deltaW += np.outer(delta_out, s[t])
            f_net = s[t] * (np.ones(self.hidden_dims) - s[t])
            delta_in = np.dot(self.W.T, delta_out) * f_net
            for bptt_step in reversed(range(max(0, t - steps), t + 1)):
                x_bptt = make_onehot(x[bptt_step], self.vocab_size)
                self.deltaV += np.outer(delta_in, x_bptt)
                self.deltaU += np.outer(delta_in, s[bptt_step - 1])
                f_net = s[bptt_step - 1] * (np.ones(self.hidden_dims) - s[bptt_step - 1])
                delta_in = np.dot(self.U.T, delta_in) * f_net


import numpy as np

# this will test the implementation of predict, acc_deltas, and acc_deltas_bptt in rnn.py, for a simple 3x2 RNN
y_exp = np.array([[ 0.39411072,  0.32179748,  0.2840918 ], [ 0.4075143,   0.32013043,  0.27235527], [ 0.41091755,  0.31606385,  0.2730186 ], [ 0.41098376,  0.31825833,  0.27075792], [ 0.41118931,  0.31812307,  0.27068762], [ 0.41356637,  0.31280332,  0.27363031], [ 0.41157736,  0.31584609,  0.27257655]])
s_exp = np.array([[ 0.66818777,  0.64565631], [ 0.80500806,  0.80655686], [ 0.85442692,  0.79322425], [ 0.84599959,  0.8270955 ], [ 0.84852462,  0.82794442], [ 0.89340731,  0.7811953 ], [ 0.86164528,  0.79916155], [ 0., 0.]])
U_exp = np.array([[ 0.89990596,  0.79983619], [ 0.5000714,   0.30009787]])
V_exp = np.array([[ 0.69787081,  0.30129314,  0.39888647], [ 0.60201076,  0.89866058,  0.70149262]])
W_exp = np.array([[ 0.57779081,  0.47890397], [ 0.22552931,  0.62294835], [ 0.39667988 , 0.19814768]])

loss_expected = 8.19118156763
loss2_expected = 3.29724981191
loss3_expected = 6.01420605985
mean_loss_expected = 1.16684249596
np_loss_expected = 0.887758278817

acc_expected = 1
acc1_np_lm_expected = 0
acc2_np_lm_expected = 1

# standard BP
deltaU_1_exp = np.array([[-0.11298744, -0.107331  ], [ 0.07341862, 0.06939134]])
deltaV_1_exp = np.array([[-0.06851441, -0.05931481, -0.05336094], [ 0.06079254,  0.0035937,   0.04875759]])
deltaW_1_exp = np.array([[-2.36320453, -2.24145091], [ 3.13861959,  2.93420307], [-0.77541506, -0.69275216]])

# BPPT
deltaU_3_exp = np.array([[-0.12007034, -0.1141893 ], [ 0.06377434, 0.06003115]])
deltaV_3_exp = np.array([[-0.07524721, -0.06495432, -0.05560471], [ 0.05465826, -0.00306904, 0.04567927]])
deltaW_3_exp = np.array([[-2.36320453, -2.24145091], [ 3.13861959,  2.93420307], [-0.77541506, -0.69275216]])

# binary prediction BP
deltaU_1_exp_np = np.array([[0.02163905, 0.01915982], [0.01045943, 0.00947443]])
deltaV_1_exp_np = np.array([[0.00223229, 0.00055869, 0.02157362], [0.00336198, 0.00047162, 0.00806956]])
deltaW_1_exp_np = np.array([[ 0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327 ]])

# binary prediction BPPT
deltaU_3_exp_np = np.array([[ 0.02163711, 0.01915766], [0.01046086, 0.00947553]])
deltaV_3_exp_np = np.array([[ 0.0022418, 0.00055819, 0.02156012], [0.00337607, 0.00047141, 0.00805541]])
deltaW_3_exp_np = np.array([[ 0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327]])

vocabsize = 3
hdim = 2
# RNN with vocab size 3 and 2 hidden layers
# Note that, for the binary prediction output vocab size should be 2 
# for test case simplicity, here we will use the same input and vocab size
r = RNN(vocabsize,hdim,vocabsize)
r.V[0][0]=0.7
r.V[0][1]=0.3
r.V[0][2]=0.4
r.V[1][0]=0.6
r.V[1][1]=0.9
r.V[1][2]=0.7

r.W[0][0]=0.6
r.W[0][1]=0.5
r.W[1][0]=0.2
r.W[1][1]=0.6
r.W[2][0]=0.4
r.W[2][1]=0.2

r.U[0][0]=0.9
r.U[0][1]=0.8
r.U[1][0]=0.5
r.U[1][1]=0.3


x = np.array([0,1,2,1,1,0,2])
d = np.array([1,2,1,1,1,1,1])
d_np = np.array([0])
d1_lm_np = np.array([2,0])
d2_lm_np = np.array([0,2])
x2 = np.array([1,1,0])
d2 = np.array([1,0,2])
x3 = np.array([1,1,2,1,2])
d3 = np.array([1,2,1,2,1])


print("### predicting y")
y,s = r.predict(x)
if not np.isclose(y_exp, y, rtol=1e-08, atol=1e-08).all():
  print("y expected\n{0}".format(y_exp))
  print("y received\n{0}".format(y))
else:
  print("y passed")
if not np.isclose(s_exp, s, rtol=1e-08, atol=1e-08).all():
  print("\ns expected\n{0}".format(s_exp))
  print("s received\n{0}".format(s))
else:
  print("s passed")

# print("\n### computing loss and mean loss")
loss = r.compute_loss(x,d)
loss2 = r.compute_loss(x2,d2)
loss3 = r.compute_loss(x3,d3)
mean_loss = r.compute_mean_loss([x,x2,x3],[d,d2,d3])
if not np.isclose(loss_expected, loss, rtol=1e-08, atol=1e-08) or not np.isclose(loss2_expected, loss2, rtol=1e-08, atol=1e-08) or not np.isclose(loss3_expected, loss3, rtol=1e-08, atol=1e-08):
  print("loss expected: {0}".format(loss_expected))
  print("loss received: {0}".format(loss))
  print("loss2 expected: {0}".format(loss2_expected))
  print("loss2 received: {0}".format(loss2))
  print("loss3 expected: {0}".format(loss3_expected))
  print("loss3 received: {0}".format(loss3))
else:
  print("loss passed")
if not np.isclose(mean_loss_expected, mean_loss, rtol=1e-08, atol=1e-08):
  print("mean loss expected: {0}".format(mean_loss_expected))
  print("mean loss received: {0}".format(mean_loss))
else:
  print("mean loss passed")


print("\n### standard BP")
r.acc_deltas(x,d,y,s)
if not np.isclose(deltaU_1_exp, r.deltaU).all():
  print("\ndeltaU expected\n{0}".format(deltaU_1_exp))
  print("deltaU received\n{0}".format(r.deltaU))
else:
  print("deltaU passed")
if not np.isclose(deltaV_1_exp, r.deltaV).all():
  print("\ndeltaV expected\n{0}".format(deltaV_1_exp))
  print("deltaV received\n{0}".format(r.deltaV))
else:
  print("deltaV passed")
if not np.isclose(deltaW_1_exp, r.deltaW).all():
  print("\ndeltaW expected\n{0}".format(deltaW_1_exp))
  print("deltaW received\n{0}".format(r.deltaW))
else:
  print("deltaW passed")

print("\n### BPTT with 3 steps")
r.deltaU.fill(0)
r.deltaV.fill(0)
r.deltaW.fill(0)

r.acc_deltas_bptt(x,d,y,s,3)
if not np.isclose(deltaU_3_exp, r.deltaU).all():
  print("\ndeltaU expected\n{0}".format(deltaU_3_exp))
  print("deltaU received\n{0}".format(r.deltaU))
else:
  print("deltaU passed")
if not np.isclose(deltaV_3_exp, r.deltaV).all():
  print("\ndeltaV expected\n{0}".format(deltaV_3_exp))
  print("deltaV received\n{0}".format(r.deltaV))
else:
  print("deltaV passed")
if not np.isclose(deltaW_3_exp, r.deltaW).all():
  print("\ndeltaW expected\n{0}".format(deltaW_3_exp))
  print("deltaW received\n{0}".format(r.deltaW))
else:
  print("deltaW passed")


# # BINARY PREDICTION TEST


# print("\n### computing binary prediction loss")
# np_loss = r.compute_loss_np(x,d_np)
# if not np.isclose(np_loss_expected, np_loss, rtol=1e-08, atol=1e-08):
#   print("np loss expected: {0}".format(np_loss_expected))
#   print("np loss received: {0}".format(np_loss))
# else:
#   print("np loss passed")

# print("\n### binary prediction BP")
# r.deltaU.fill(0)
# r.deltaV.fill(0)
# r.deltaW.fill(0)

# r.acc_deltas_np(x,d_np,y,s)
# if not np.isclose(deltaU_1_exp_np, r.deltaU).all():
#   print("\ndeltaU expected\n{0}".format(deltaU_1_exp_np))
#   print("deltaU received\n{0}".format(r.deltaU))
# else:
#   print("deltaU passed")
# if not np.isclose(deltaV_1_exp_np, r.deltaV).all():
#   print("\ndeltaV expected\n{0}".format(deltaV_1_exp_np))
#   print("deltaV received\n{0}".format(r.deltaV))
# else:
#   print("deltaV passed")
# if not np.isclose(deltaW_1_exp_np, r.deltaW).all():
#   print("\ndeltaW expected\n{0}".format(deltaW_1_exp_np))
#   print("deltaW received\n{0}".format(r.deltaW))
# else:
#   print("deltaW passed")


# print("\n### binary prediction BPTT with 3 steps")
# r.deltaU.fill(0)
# r.deltaV.fill(0)
# r.deltaW.fill(0)

# r.acc_deltas_bptt_np(x,d_np,y,s,3)
# if not np.isclose(deltaU_3_exp_np, r.deltaU).all():
#   print("\ndeltaU expected\n{0}".format(deltaU_3_exp_np))
#   print("deltaU received\n{0}".format(r.deltaU))
# else:
#   print("deltaU passed")
# if not np.isclose(deltaV_3_exp_np, r.deltaV).all():
#   print("\ndeltaV expected\n{0}".format(deltaV_3_exp_np))
#   print("deltaV received\n{0}".format(r.deltaV))
# else:
#   print("deltaV passed")
# if not np.isclose(deltaW_3_exp_np, r.deltaW).all():
#   print("\ndeltaW expected\n{0}".format(deltaW_3_exp_np))
#   print("deltaW received\n{0}".format(r.deltaW))
# else:
#   print("deltaW passed")


# print("\n### compute accuracy for binary prediction")
# acc = r.compute_acc_np(x, d_np)
# if acc != acc_expected:
#   print("acc expected\n{0}".format(acc_expected))
#   print("acc received\n{0}".format(acc))
# else:
#   print("acc passed")


# print("\n### compute accuracy for LM binary prediction")
# acc1 = r.compare_num_pred(x, d1_lm_np)
# acc2 = r.compare_num_pred(x, d2_lm_np)
# if acc1 != acc1_np_lm_expected or acc2 != acc2_np_lm_expected:
#   print("LM acc1 expected\n{0}".format(acc1_np_lm_expected))
#   print("LM acc1 received\n{0}".format(acc1))
#   print("LM acc2 expected\n{0}".format(acc2_np_lm_expected))
#   print("LM acc2 received\n{0}".format(acc2))
# else:
#   print("LM acc passed")
