import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    print(predictions.shape)
    predictions -= np.max(predictions)
    ei = np.exp(predictions)
    denominator = np.sum(ei)
    res = ei / denominator
    return res
    # raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    true_vals = np.zeros(probs.shape)
    # print('true_vals: ', true_vals, 'shape: ', true_vals.shape)
    # print('probs', probs)
    # print('target_index', target_index, 'shape', target_index.shape)
    # print("t ", true_vals[(0, 0)], true_vals[(0, 1)])
    for i in range(len(target_index)):
        true_vals[(i, target_index)] = 1
    # true_vals[target_index] = 1
    res = np.sum(true_vals*np.log(probs))
    print('true_vals:', true_vals)
    print('cross_entropy_loss: ', res)
    return res
    
    
def dSk_dyk(y, k):
    s_k =  np.exp(y[k]) / np.sum(np.exp(y))
    return s_k - s_k**2
    
    
def dSi_dyk(y, i, k):
    s = lambda y, k: np.exp(y[k]) / np.sum(np.exp(y))
    s_k = s(y, k)
    s_i = s(y, i)
    return s_i * s_k
    
    
def softmax_with_gradient(y):
    s = softmax(y)
    return s, s - np.square(s)


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    #predictions is an array of 'y'
    single = (predictions.ndim == 1)
    if single:
        predictions = predictions.reshape(1, predictions.shape[0])
        target_index = np.array([target_index])
    
    sf = softmax(predictions)                       # S
    # print('predictions: ', predictions)
    # print('softmax: ', sf)
    # print('target_index: ', target_index)
    loss = cross_entropy_loss(sf, target_index)     # L

    indicator = np.zeros(sf.shape)
    indicator[np.arange(sf.shape[0]), target_index] = 1     # 1(y)
    dprediction = (sf - indicator) / predictions.shape[0]   # dL/dZ = (S - 1(y)) / N

    if single:
        dprediction = dprediction.reshape(dprediction.size)
    
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
