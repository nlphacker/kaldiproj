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

import json

from io_func.pfile_io import read_data_args, read_dataset

from io_func.model_io import _nnet2file, _file2nnet, _cnn2file, _file2cnn, log

from utils.learn_rates import LearningRateExpDecay
from utils.utils import parse_conv_spec, parse_lrate, parse_arguments, parse_activation, activation_to_txt, string_2_bool

from io_func.convert2kaldi import _nnet2kaldi

from models.cnn_sat import CNN_SAT

if __name__ == '__main__':

    import sys

    arg_elements=[]
    for i in range(1, len(sys.argv)):
        arg_elements.append(sys.argv[i])
    arguments = parse_arguments(arg_elements) 

    if (not arguments.has_key('train_data')) or (not arguments.has_key('valid_data')) or (not arguments.has_key('conv_nnet_spec')) or (not arguments.has_key('full_nnet_spec')) or (not arguments.has_key('wdir')) or (not arguments.has_key('conv_output_file')) or (not arguments.has_key('full_output_file')) or (not arguments.has_key('ivec_output_file')) or (not arguments.has_key('update_part')) or (not arguments.has_key('ivec_dim')):
        print "Error: the mandatory arguments are: --train-data --valid-data --conv-nnet-spec --full-nnet-spec --wdir --conv-output-file --full-output-file"
        exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    conv_nnet_spec = arguments['conv_nnet_spec']
    full_nnet_spec = arguments['full_nnet_spec']
    ivec_nnet_spec = arguments['ivec_nnet_spec']
    wdir = arguments['wdir']
    conv_output_file = arguments['conv_output_file']
    full_output_file = arguments['full_output_file']
    ivec_output_file = arguments['ivec_output_file']

    # output format: kaldi or janus
    output_format = 'kaldi'
    if arguments.has_key('output_format'):
        output_format = arguments['output_format']
    if output_format != 'kaldi':
        print "Error: the output format only supports Kaldi"
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

    # batch_size and momentum
    batch_size=256
    momentum=0.5
    if arguments.has_key('batch_size'):
        batch_size = int(arguments['batch_size'])
    if arguments.has_key('momentum'):
        momentum = float(arguments['momentum'])

    # conv layer configuraitons
    conv_layer_configs = parse_conv_spec(conv_nnet_spec, batch_size)
    # full layer configurations
    nnet_layers = full_nnet_spec.split(":")
    hidden_layers_sizes = []
    for i in range(0, len(nnet_layers)-1):
        hidden_layers_sizes.append(int(nnet_layers[i]))
    n_outs = int(nnet_layers[-1])
    # ivec layer configurations
    nnet_layers = ivec_nnet_spec.split(":")
    ivec_layers_sizes = []
    for i in xrange(len(nnet_layers)):
        ivec_layers_sizes.append(int(nnet_layers[i]))

    conv_activation = T.nnet.sigmoid
    full_activation = T.nnet.sigmoid
    if arguments.has_key('conv_activation'):
        conv_activation = parse_activation(arguments['conv_activation'])
    if arguments.has_key('full_activation'):
        full_activation = parse_activation(arguments['full_activation'])

    # which part of the network to be updated
    update_part = []
    for part in arguments['update_part'].split(':'):
        update_part.append(int(part))

    # the dimension of the ivectors
    ivec_dim = int(arguments['ivec_dim'])

    # whether to use the fast version of CNN with pylearn2
    use_fast = False
    if arguments.has_key('use_fast'):
        use_fast = string_2_bool(arguments['use_fast'])        
 
    train_dataset, train_dataset_args = read_data_args(train_data_spec)
    valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)
    
    # reading data 
    train_sets, train_xy, train_x, train_y = read_dataset(train_dataset, train_dataset_args)
    valid_sets, valid_xy, valid_x, valid_y = read_dataset(valid_dataset, valid_dataset_args)

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # construct the cnn architecture
    cnn = CNN_SAT(numpy_rng=numpy_rng, theano_rng = theano_rng,
              batch_size = batch_size, n_outs=n_outs,
              conv_layer_configs = conv_layer_configs,
              hidden_layers_sizes = hidden_layers_sizes,
              ivec_layers_sizes = ivec_layers_sizes,
              conv_activation = conv_activation, 
              full_activation = full_activation,
              use_fast = use_fast, update_part = update_part, ivec_dim = ivec_dim)

    if arguments.has_key('conv_input_file'):
        _file2cnn(cnn.conv_layers, filename=arguments['conv_input_file'])
    if arguments.has_key('full_input_file'):
        _file2nnet(cnn.full_layers, filename = arguments['full_input_file'])
    if arguments.has_key('ivec_input_file'):
        _file2nnet(cnn.ivec_layers, set_layer_num = len(ivec_layers_sizes) + 1, filename = arguments['ivec_input_file'], withfinal=False)

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = cnn.build_finetune_functions(
                (train_x, train_y), (valid_x, valid_y),
                batch_size=batch_size)

    log('> ... finetunning the model')
    start_time = time.clock()

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

        log('> epoch %d, smallest lrate %f' % (lrate.epoch, lrate.lowest_error))
	lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

    # output conv layer config
#    for i in xrange(len(conv_layer_configs)):
#        conv_layer_configs[i]['activation'] = activation_to_txt(conv_activation)
#        with open(wdir + '/conv.config.' + str(i), 'wb') as fp:
#            json.dump(conv_layer_configs[i], fp, indent=2, sort_keys = True)
#            fp.flush()

    # output the conv part
    _cnn2file(cnn.conv_layers, filename=conv_output_file)
    # output the full part
    _nnet2file(cnn.full_layers, filename=full_output_file)
    _nnet2file(cnn.ivec_layers, set_layer_num = len(ivec_layers_sizes) + 1, filename=ivec_output_file, withfinal=False)
#    _nnet2kaldi(str(cnn.conv_output_dim) + ':' + full_nnet_spec, filein = wdir + '/nnet.finetune.tmp', fileout = full_output_file)

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

