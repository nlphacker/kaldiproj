#!/bin/bash

# Copyright 2012 Vassil Panayotov

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

#RM1_ROOT=./test1
source path.sh

# Download and extract CMU's feature files
mkdir -p $RM1_ROOT
wget -P $RM1_ROOT http://www.speech.cs.cmu.edu/databases/rm1/rm1_cepstra.tar.gz
tar -C $RM1_ROOT/ -xf $RM1_ROOT/rm1_cepstra.tar.gz

# Download the available LDC metadata
# For some reason wget needs to be run twice in order to get all needed data ...
wget -P $RM1_ROOT -mk --no-parent -r -c -v -nH http://www.ldc.upenn.edu/Catalog/docs/LDC93S3B/
wget -P $RM1_ROOT -mk --no-parent -r -c -v -nH http://www.ldc.upenn.edu/Catalog/docs/LDC93S3B/
mv $RM1_ROOT/Catalog/docs/LDC93S3B $RM1_ROOT/
rm -rf $RM1_ROOT/Catalog