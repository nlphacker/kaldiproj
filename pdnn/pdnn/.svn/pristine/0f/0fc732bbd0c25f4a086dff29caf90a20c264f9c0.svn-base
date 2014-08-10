# Copyright 2013    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.dnn_sat import DNN_SAT
from models.dropout_nnet import DNN_Dropout

from io_func.pfile_io import read_data_args, read_dataset

from io_func.model_io import _nnet2file, _file2nnet, log
from utils.utils import parse_arguments, parse_lrate, parse_activation, parse_two_integers
from utils.learn_rates import LearningRateExpDecay

from io_func.convert2kaldi import _nnet2kaldi, _nnet2kaldi_maxout
from io_func.convert2janus import _nnet2janus, _nnet2janus_maxout


if __name__ == '__main__':

    import sys

    arg_elements=[]
    for i in range(1, len(sys.argv)):
        arg_elements.append(sys.argv[i])
    arguments = parse_arguments(arg_elements) 

    if (not arguments.has_key('train_data')) or (not arguments.has_key('valid_data')) or (not arguments.has_key('nnet_spec')) or (not arguments.has_key('wdir')) or (not arguments.has_key('output_file')) or (not arguments.has_key('ivec_nnet_spec')) or (not arguments.has_key('ivec_output_file')) or (not arguments.has_key('si_model')):
        print "Error: the mandatory arguments are: --train-data --valid-data --nnet-spec --wdir --output-file --ivec-nnet-spec --ivec-output-file -si-model"
        exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']
    output_file = arguments['output_file']
    ivec_nnet_spec = arguments['ivec_nnet_spec']
    ivec_output_file = arguments['ivec_output_file']
    si_model_file = arguments['si_model']

    # regularization for hidden layer parameter
    max_col_norm = None
    l1_reg = None
    l2_reg = None
    if arguments.has_key('max_col_norm'):
        max_col_norm = float(arguments['max_col_norm'])
    if arguments.has_key('l1_reg'):
        l1_reg = float(arguments['l1_reg'])
    if arguments.has_key('l2_reg'):
        l2_reg = float(arguments['l2_reg'])

    # output format: kaldi or janus
    output_format = 'kaldi'
    if arguments.has_key('output_format'):
        output_format = arguments['output_format']
    if output_format != 'kaldi' and output_format != 'janus':
        print "Error: the output format only supports Kaldi and Janus"
        exit(1)

    # learning rate
    if not arguments.has_key('lrate'):
        lrate = LearningRateExpDecay(start_rate=0.08,
                                 scale_by = 0.5,
                                 min_derror_decay_start = 0.05,
                                 min_derror_stop = 0.05,
                                 min_epoch_decay_start=15)
    else:
        lrate = parse_lrate(arguments['lrate']) 
    if lrate is None:
        print "Error: lrate object is None"
        exit(1)

    # batch_size and momentum
    batch_size=256
    momentum=0.5
    if arguments.has_key('batch_size'):
        batch_size = int(arguments['batch_size'])
    if arguments.has_key('momentum'):
        momentum = float(arguments['momentum'])
    # parse the iVecNN and DNN architecture
    nnet_layers = nnet_spec.split(":")
    n_ins = int(nnet_layers[0])
    hidden_layers_sizes = []
    for i in range(1, len(nnet_layers)-1):
        hidden_layers_sizes.append(int(nnet_layers[i]))
    n_outs = int(nnet_layers[-1])

    ivec_nnet_layers = ivec_nnet_spec.split(":")
    ivec_dim = int(ivec_nnet_layers[0])
    ivec_layers_sizes = []
    for i in range(1, len(ivec_nnet_layers)-1):
        ivec_layers_sizes.append(int(ivec_nnet_layers[i]))
    
    # parse the activation function
    activation = T.nnet.sigmoid
    do_maxout = False
    pool_size = 1
    do_pnorm = False
    pnorm_order = 1
    if arguments.has_key('activation'):
        activation = parse_activation(arguments['activation'])
        if arguments['activation'].startswith('maxout'):
            do_maxout = True
            pool_size = int(arguments['activation'].replace('maxout:',''))
        elif arguments['activation'].startswith('pnorm'):
            do_pnorm = True
            pool_size, pnorm_order = parse_two_integers(arguments['activation'])

    train_dataset, train_dataset_args = read_data_args(train_data_spec)
    valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)
    train_sets, train_xy, train_x, train_y = read_dataset(train_dataset, train_dataset_args)
    valid_sets, valid_xy, valid_x, valid_y = read_dataset(valid_dataset, valid_dataset_args)

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # doesn't deal with dropout 
    dnn = DNN_SAT(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
              activation = activation, do_maxout = do_maxout, pool_size = pool_size,
              do_pnorm = do_pnorm, pnorm_order = pnorm_order,
              max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg,
              ivec_dim = ivec_dim, ivec_layers_sizes = ivec_layers_sizes)
    # read the initial DNN 
    _file2nnet(dnn.sigmoid_layers, filename = si_model_file)

    # get the training, validation and testing function for iVecNN
    dnn.params = dnn.ivec_params
    dnn.delta_params = dnn.ivec_delta_params
    log('> ... getting the finetuning functions for iVecNN')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (train_x, train_y), (valid_x, valid_y),
                batch_size=batch_size)

    log('> ... learning the iVecNN network')
    while (lrate.get_rate() != 0):
        train_error = []
        while (not train_sets.is_finish()):
            train_sets.load_next_partition(train_xy)
            for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
        train_sets.initialize_read()
        log('> epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))

        valid_error = []
        while (not valid_sets.is_finish()):
            valid_sets.load_next_partition(valid_xy)
            for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
                valid_error.append(valid_fn(index=batch_index))
        valid_sets.initialize_read()
        log('> epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))

        lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

    # get the training, validation and testing function for DNN
    dnn.params = dnn.sigmoid_params
    dnn.delta_params = dnn.sigmoid_delta_params
    log('> ... getting the finetuning functions for DNN')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (train_x, train_y), (valid_x, valid_y),
                batch_size=batch_size)

    # initialize the learning rate schedule 
    if not arguments.has_key('lrate'):
        lrate = LearningRateExpDecay(start_rate=0.08,
                                 scale_by = 0.5,
                                 min_derror_decay_start = 0.05,
                                 min_derror_stop = 0.05,
                                 min_epoch_decay_start=15)
    else:
        lrate = parse_lrate(arguments['lrate'])

    log('> ... learning the DNN model in the new feature space')
    while (lrate.get_rate() != 0):
        train_error = []
        while (not train_sets.is_finish()):
            train_sets.load_next_partition(train_xy)
            for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
        train_sets.initialize_read()
        log('> epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))

        valid_error = []
        while (not valid_sets.is_finish()):
            valid_sets.load_next_partition(valid_xy)
            for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
                valid_error.append(valid_fn(index=batch_index))
        valid_sets.initialize_read()
        log('> epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))

        lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

    # output both iVecNN and DNN
    _nnet2file(dnn.ivec_layers, set_layer_num = len(ivec_nnet_layers)-1, filename=wdir + '/ivec.finetune.tmp', withfinal=True)
    _nnet2file(dnn.sigmoid_layers, filename=wdir + '/nnet.finetune.tmp')

    # determine whether it's BNF based on layer sizes
    set_layer_num = -1
    withfinal = True
    bnf_layer_index = 1
    while bnf_layer_index < len(hidden_layers_sizes):
        if hidden_layers_sizes[bnf_layer_index] < hidden_layers_sizes[bnf_layer_index - 1]:  
            break
        bnf_layer_index = bnf_layer_index + 1

    if bnf_layer_index < len(hidden_layers_sizes):  # is bottleneck
        set_layer_num = bnf_layer_index+1
        withfinal = False

    # finally convert the nnet to kaldi or janus format
    if do_maxout:
      _nnet2kaldi_maxout(nnet_spec, pool_size = pool_size, set_layer_num = set_layer_num, filein = wdir + '/nnet.finetune.tmp', fileout = output_file, withfinal=withfinal)
    else:
      _nnet2kaldi(nnet_spec, set_layer_num = set_layer_num, filein = wdir + '/nnet.finetune.tmp', fileout = output_file, withfinal=withfinal)
    _nnet2kaldi(ivec_nnet_spec, set_layer_num = len(ivec_nnet_layers)-1, filein = wdir + '/ivec.finetune.tmp', fileout = ivec_output_file, withfinal=False)
