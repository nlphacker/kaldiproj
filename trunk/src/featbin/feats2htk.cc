// featbin/feats2htk.cc

// Copyright 2013   Petr Motlicek

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-common.h"
#include "matrix/matrix-lib.h"

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;


    const char *usage =
        "Sve features as HTK files:\n" 
        "[ Each Utterance will be stored as a unique HTK file ]\n"
        "Usage: feats2htk [options] in-rspecifier\n";

    ParseOptions po(usage);
    std::string dir_out = "./";
    std::string ext_out = "fea";
    int32 SamplePeriod = 10000;
    int32 SampleKind = 9; //USER  
    /*
    0 WAVEFORM sampled waveform
    1 LPC linear prediction filter coefficients
    2 LPREFC linear prediction reflection coefficients
    3 LPCEPSTRA LPC cepstral coefficients
    4 LPDELCEP LPC cepstra plus delta coefficients
    5 IREFC LPC reflection coef in 16 bit integer format
    6 MFCC mel-frequency cepstral coefficients
    7 FBANK log mel-filter bank channel outputs
    8 MELSPEC linear mel-filter bank channel outputs
    9 USER user defined sample kind
    10 DISCRETE vector quantised data
    11 PLP PLP cepstral coefficients
    */

    po.Register("output-ext", &ext_out, "Output ext of HTK files");
    po.Register("output-dir", &dir_out, "Output directory");
    po.Register("SamplePeriod", &SamplePeriod, "HTK sampPeriod - sample period in 100ns units");
    po.Register("SampleKind", &SampleKind, "HTK parmKind - a code indicating the sample kind");



    po.Read(argc, argv);

    //std::cout << "Dir: " << dir_out << " ext: " << ext_out << "\n"; 

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);

    // check or create output dir:
    const char * c = dir_out.c_str();
   if ( access( c, 0 ) != 0 ){
    if (mkdir(c, S_IRWXU|S_IRGRP|S_IXGRP) != 0)
       KALDI_ERR << "Could not create output directory: " << dir_out;
   /*
    else if (chdir(c) != 0)
       KALDI_ERR << "first chdir() error: " << dir_out;
    else if (chdir("..") != 0)
       KALDI_ERR << "second chdir() error: " << dir_out;
    else if (rmdir(c) != 0)
       KALDI_ERR << "rmdir() error: " << dir_out;
   */
    }


    // HTK parameters
    HtkHeader hdr;
    hdr.mSamplePeriod = SamplePeriod;
    hdr.mSampleKind = SampleKind;

    
    // write to the HTK files
    int32 num_frames, dim, count=0;
    SequentialBaseFloatMatrixReader feats_reader(rspecifier);
    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string utt = feats_reader.Key();
      const Matrix<BaseFloat> &feats = feats_reader.Value();
      num_frames = feats.NumRows(), dim = feats.NumCols();
      //std::cout << "Utt: " << utt<< " Frames: " << num_frames << " Dim: " << dim << "\n";

      hdr.mNSamples = num_frames;
      hdr.mSampleSize = sizeof(float)*dim;

      Matrix<BaseFloat> output(num_frames, dim, kUndefined);
      std::stringstream ss;
      ss << dir_out << "/" << utt << "." << ext_out; 
      output.Range(0, num_frames, 0, dim).CopyFromMat(feats.Range(0, num_frames, 0, dim));	
      std::ofstream os(ss.str().c_str(), std::ios::out|std::ios::binary);
      WriteHtk(os, output, hdr);  
      count++;    
    }
    std::cout << count << " feature files generated in the firecory: " << dir_out <<"\n";
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


