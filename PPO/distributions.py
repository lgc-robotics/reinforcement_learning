import tensorflow as tf
import numpy as np

EPS=1E-10

class DiagonalGaussian():
    def __init__(self,mean,std,log_std):
        self.mean=mean
        self.std=std
        self.log_std=log_std
    
    # return shape: (batch,)
    def log_likelihood(self,x):
        pre_sum = -0.5 * (((x-self.mean)/(tf.exp(self.log_std)+EPS))**2 + 2*self.log_std + tf.math.log(tf.constant(2*np.pi)))
        return tf.reduce_sum(pre_sum, axis=1)
    
    def sample(self):
        return self.mean+tf.random.normal(tf.shape(self.mean))*self.std
    
    def entropy(self):
        pass   
