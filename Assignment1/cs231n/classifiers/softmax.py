import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W)
  train_size = X.shape[0]
  classes_cnt = W.shape[1]

  for i in range(train_size):
    f = S[i] - np.max(S[i])
    sm = np.exp(f) / np.sum(np.exp(f))
    loss += -np.log(sm[y[i]])

    for j in range(classes_cnt):
      dW[:, j] += X[i] * sm[j]

    dW[:, y[i]] -= X[i]

  loss /= train_size
  dW /= train_size

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W)
  train_size = X.shape[0]
  S -= np.max(S, axis=1, keepdims=True)

  sum_exp_S = np.exp(S).sum(axis=1, keepdims=True)
  sm_matrix = np.exp(S) / sum_exp_S

  loss = np.sum(-np.log(sm_matrix[np.arange(train_size), y]))

  sm_matrix[np.arange(train_size), y] -= 1
  dW = X.T.dot(sm_matrix)

  loss /= train_size
  dW /= train_size

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

