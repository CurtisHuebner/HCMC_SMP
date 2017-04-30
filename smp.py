#!/usr/bin/env python3
import pandas as pa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch as t
import torch.nn.functional as f
from torch.autograd import Variable

import argparse
import time
import math

N = 10000 #default number of hcmc iterations

def load_file(path):
    """Loads a csv file from a path as a pandas array.
    
    (Larger Description)

    Args:
        path: The path to the file
    
    Returns:
        A pandas dataframe

    Raises:
    """
    return pa.read_csv(path)

def preprocess_csv(data_frame):
    """Turn the csv file of stock prices into normalized log deltas.

    (Larger Description)

    Args:
        data_frame: The data to preprocess
    
    Returns:
        A new data frame with normalised log values,
        The means of the dataset,
        The variances of the dataset.

    Raises:None

    """
    #TODO: make nan filter
    array = data_frame.as_matrix()
    #flip order of dates
    array = np.flip(array,0)
    #discard dates and convert to floats
    array = array[:,1:].astype(np.float32)
    #take log differences
    array = (np.log(array[1:,:])-np.log(array[:-1,:]))
    #replace not numbers with zeros
    array[np.where(np.logical_not(np.isfinite(array)))] = 0
    std_dev = np.std(array,0)
    mean = np.mean(array,0)
    array = (array - mean)/std_dev
    return array,mean,std_dev

class HCMCSampler:
    """updates the model parameters by preforming n hcmc updates

    (larger description)
    """
    def __init__(self,model,data,nllh):
        self.model = model
        self.data = data
        self.nllh = nllh

    def resample_r(self):
        p_list = self.model.parameter_list()
        self.r_list = []
        for param in p_list:
            s = param.size()
            means = t.zeros(s)
            self.r_list.append(t.normal(means,1).cuda())#todo verify that this is correct
    def data_pass(self):
        p_list = self.model.parameter_list()

        def zero_grad(x):
            if x.grad is not None:
                x.grad.data.zero_()
        list(map(zero_grad,p_list))
        output = self.model.inference(self.data)
        loss = self.nllh(output,self.data)
        loss.backward()
        g_list = list(map((lambda x: x.grad.data),p_list))
#       for grad in g_list:
#           print(key,t.sum(grad))
        return g_list,loss

    def step(self,epsilon,n=1):
        self.resample_r()
        p_list = [x.data for x in self.model.parameter_list()]
        r_list = self.r_list
        def assign(x,y):
            x.data = y
        for i in range(n):
            if (np.random.randn() > 0):
                epsilon = -epsilon
            #TODO: Clean up implementation with getter and setters
            g_list,_ = self.data_pass()
            r_list = list(map(lambda x,y: x-y*epsilon/2,r_list,g_list))

            p_list = list(map(lambda x,y: x+y*epsilon,p_list,r_list))
            list(map(assign,self.model.parameter_list(),p_list))

            g_list,loss = self.data_pass()
            r_list = list(map(lambda x,y: x-y*epsilon/2,r_list,g_list))
        

def total_nllh(output,data):
    """Compute the log likelyhood of the data given the model

    (Larger Description)

    Args:
    
    Returns:

    Raises:
    """
    data = Variable(t.FloatTensor(data).cuda())
    means,log_std_devs = output
    variance = t.exp(2*log_std_devs)
    means = means
    #TODO: check math here
    deviation = (data - means)*(data - means)/(2*variance) + math.log(2*math.pi)+t.log(variance)/2

    return t.sum(deviation)


def validate_model(model,validation_set):
    """Validate the correctness of the model

    (Larger Description)

    Args:
    
    Returns:

    Raises:
    """
    pass

class ParameterGroup:
    
    def __init__(self,parameter_dict):
        self.parameters = parameter_dict

    def get_prior_llh(self):
        prior = 0
        for value in self.parameters.values():
            prior += value.get_prior_llh()

    def parameter_list(self):
        p_list = []
        for value in self.parameters.values():
            p_list += value.parameter_list()
        return p_list

    def cuda(self):
        for value in self.parameters.values():
            value.cuda()

    def cpu(self):
        for value in self.parameters.values():
            value.cpu()

    def __getitem__(self,key):
        return self.parameters[key]

class TensorParameter:
    def __init__(self,shape,std_dev):
        self.parameter = Variable(t.FloatTensor(np.random.normal(size=shape,
            scale=std_dev)).cuda(),requires_grad = True)
        self.var = std_dev*std_dev
        self.shape = shape
    
    def parameter_list(self):
        return [self.parameter]
    
    def val(self):
        return self.parameter
    
    def cuda(self):
        self.parameter.cuda()

    def cpu(self):
        self.parameter.cpu()

    def get_prior_llh(self,dims):
        prob = -t.log(self.parameter)**2 \
                /(2*self.var)-t.log(2*math.pi)-t.log(self.var)/2
        for dim in dims:
            prob = sum(prob,dim)

        return t.squeeze(prob)

class Model(ParameterGroup):
    """Stock market model

    """
    def __init__(self,batch_size,in_dim,h_dim,out_dim = None,std_dev = 1):
        if out_dim is None:
            out_dim = in_dim
        p_dict = {}
        fx =  lambda n,m : TensorParameter((batch_size,n,m,),std_dev/math.sqrt(m))
        p_dict['itoh'] = fx(in_dim,h_dim)
        p_dict['htoh'] = fx(h_dim,h_dim)
        p_dict['htos'] = fx(h_dim,out_dim)
        p_dict['htom'] = fx(h_dim,out_dim)
        p_dict['initial_state'] = fx(1,h_dim)
        super().__init__(p_dict)

    def inference(self,data):
        f,m,s = self.batch_rnn_layer(self,data)
        return m,s

    def get_prior_llh(self):
        return self.get_prior_llh()

    def batch_rnn_layer(self,batch_parameters,inputs,state = None):
        """Computes the output of an rnn layer over the model

        (Larger Description)

        Args:
            initial_states: Initial states to the rnn
            batch_parameters: A tuple of batches of parameters
            inputs:Inputs to the neural network in the form 
            of a b*n*7 numpy array
        
        Returns:
            The final state of the RNN, along with it's outputs.
            Outputs are a tuple of gaussian means and standard deviations.

        Raises:None
        """
        loop_iterations = inputs.shape[1]
        t_inputs = Variable(t.FloatTensor(inputs).cuda())
        output_mean = Variable(t.FloatTensor(np.zeros_like(inputs)).cuda())
        output_lsd =Variable(t.FloatTensor(np.zeros_like(inputs)).cuda())
        current_states=batch_parameters['initial_state'].val()
        if state is None:
            current_states=batch_parameters['initial_state'].val()
        else:
            current_states=states

        for i in range(loop_iterations):
            current_states,output_mean[:,i,:],output_lsd[:,i,:] = \
                    self.batch_rnn_cell(
                            current_states,batch_parameters,t_inputs[:,i,:])
        final_states = current_states

        return final_states,output_mean,output_lsd

    def batch_rnn_cell(self,states,batch_parameters,inputs):
        """Computes one timestep of a batch rnn

        (Larger Description)

        Args:
            initial_states: Initial states to the rnn
            batch_parameters: A tuple of batches of parameters
            inputs:Inputs to the neural network in the form 
            of a b*n*7 numpy array
        
        Returns:
            The final state of the RNN, along with it's outputs.
            Outputs are a tuple of gaussian means and standard deviations.

        Raises:None
        """
        #Simple RNN for now
        htos = batch_parameters['htos'].val()
        htom = batch_parameters['htom'].val()
        ith = batch_parameters['itoh'].val()
        hth = batch_parameters['htoh'].val()
        inputs = inputs[:,None,:]

        output_mean = t.bmm(states,htom)
        output_lsd = t.bmm(states,htos)
        #lets use sines 'cause why not
        next_states = (t.tanh(t.bmm(inputs,ith))+t.tanh(t.bmm(states,hth)))
        #normalize the hidden state vector
        next_states = next_states
        return next_states,output_mean,output_lsd



def get_args():
    parser = argparse.ArgumentParser(description=
        "Train a model on some csv data")
    parser.add_argument("path")
    parser.add_argument("-n")
    return parser.parse_args()

def main():
    """

    (Larger Description)

    Args:
    
    Returns:

    Raises:
    """
    BS = 100 
    #pars args
    args = get_args()
    if args.n is None:
        args.n = N
    else:
        args.n = int(args.n)

    #process data
    data_frame = load_file(args.path)
    data,means,stdevs = preprocess_csv(data_frame)

    in_dim = data.shape[-1]
    data_set = np.tile(data,(BS,1,1))
    data = data_set[:,-1600:-200,:]
    test_data = data_set[:,-199:,:]


    #train model
    model = Model(BS,in_dim,20)
    sampler = HCMCSampler(model,data,total_nllh)
    optimizer = t.optim.RMSprop(model.parameter_list(),lr=0.0002)
    uptick = 0
    for i in range(3000):
        optimizer.step()
        _,loss = sampler.data_pass()
        if i is 0:
            old_val_loss = loss
        if i % 5 is 0:
            new_val_loss = validate_model(model,data,test_data,total_nllh)
            if (new_val_loss > old_val_loss and new_val_loss < 0).data.cpu().numpy()[0]:
                uptick += 1
            else: 
                uptick = 0
            if uptick is 2:
                break

            old_val_loss = new_val_loss

    for i in range(100):
        sampler.step(0.0001,n=10)
        _,loss = sampler.data_pass()

    validate_model(model,data,test_data,total_nllh)

def validate_model(model,data,test_data,nllh):
    f,_,_ = model.batch_rnn_layer(model,data)
    _,m,s = model.batch_rnn_layer(model,test_data)
    output = m,s
    loss1 = nllh(output,test_data)
    output = [Variable(t.zeros(x.size()).cuda()) for x in output]
    loss2 = nllh(output,test_data)
    print((loss1-loss2))
    return loss1-loss2

def test_hcmc():
    pass



if __name__ == "__main__":
    main()
