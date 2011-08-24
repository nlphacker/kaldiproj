// rnn/rnn-rescore.cc

// Copyright 2009-2011  Stefan Kombrink 

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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <float.h>

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fst/fst.h"
#include "fst/mutable-fst.h"
#include "fst/queue.h"
#include "fst/rmepsilon.h"
#include "fst/dfs-visit.h"
#include "lat/kaldi-lattice.h"

using namespace std;
using namespace kaldi;

typedef kaldi::Matrix<BaseFloat> KaldiMatrix;
typedef kaldi::Vector<BaseFloat> KaldiVector;

class RNN{
  public:
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;

  private:
    KaldiMatrix V1_,U1_,W2_,Cl_; // weights
    KaldiVector b1_,b2_,cl_b_;   // biases TODO not yet used!!!
    KaldiVector h_,cl_,y_;       // activations

    map<int32,string> int2word_;                  // maps from rnn ints to word strings
    map<string,int32> word2int_;                  // maps from word strings to rnn ints
    map<int32,int32> intlat2intrnn;               // maps from ints in lattices to ints in RNN
    map<int32,int32> intrnn2intlat;               // maps from ints in RNN to ins in lattices
    vector<int32> int2class_;                     // mapping words (integer) to their classes
    map<int32,int32> class2minint_,class2maxint_; // determines the range of a class of words in the output layer

    // others...
    BaseFloat OOVPenalty_;
    BaseFloat Lambda_;
    const fst::SymbolTable* wordsym_;

    int32 ivcnt_;
    int32 oovcnt_;

  public:
    inline int32 VocabSize() const { return V1_.NumRows(); }
    inline int32 HiddenSize() const { return V1_.NumCols(); }
    inline int32 ClassSize() const { return Cl_.NumCols(); }

    inline int32 IVProcessed() const { return ivcnt_; }
    inline int32 OOVProcessed() const { return oovcnt_; }

    inline bool IsIV(int32 w) const { return w>=0; }
    inline bool IsOOV(int32 w) const { return !IsIV(w); }

    BaseFloat OOVPenalty() const { return OOVPenalty_; }
    void SetOOVPenalty( BaseFloat oovp ) { OOVPenalty_=oovp; }

    BaseFloat Lambda() const { return Lambda_; }
    void SetLambda(BaseFloat l) { Lambda_=l; }

    void Read(istream& in, bool binary) {
      ExpectMarker(in,binary,"<rnnlm_v2.0>");
      ExpectMarker(in,binary,"<v1>"); in >> V1_;
      ExpectMarker(in,binary,"<u1>"); in >> U1_;
      ExpectMarker(in,binary,"<b1>"); in >> b1_;
      ExpectMarker(in,binary,"<w2>"); in >> W2_;
      ExpectMarker(in,binary,"<b2>"); in >> b2_;
      ExpectMarker(in,binary,"<cl>"); in >> Cl_;
      ExpectMarker(in,binary,"<cl_b>"); in >> cl_b_;
      ExpectMarker(in,binary,"<classes>");
      ReadIntegerVector(in,binary,&int2class_);

     // determine range for classes 
     // THIS ASSUMES CLASSES ARE ENUMERABLE AND INCREASING!!
     
      int32 cl;
      for (int32 i=0;i<VocabSize();i++) {
        cl=int2class_[i];
        if (class2minint_.count(cl)==0) class2minint_[cl]=i; // mapping class -> start int
        class2maxint_[cl]=i; // mapping class -> max int
      }

      ExpectMarker(in,binary,"<words>");
      std::string wrd;

      // read vocabulary
      for (int32 i=0;i<VocabSize();i++) {
        ReadMarker(in,binary,&wrd);
        word2int_[wrd]=i;
        int2word_[i]=wrd;
      }
  
      // prepare activation layers
      h_.Resize(HiddenSize(),kSetZero);
      y_.Resize(VocabSize(),kSetZero);
      cl_.Resize(ClassSize(),kSetZero);       
    };

    // this is required to convert openfst symbol IDs to RNN IDs
    void SetLatticeSymbols(const fst::SymbolTable& symtab) {
        wordsym_=&symtab;
        int32 j,ii,oov=-1;
        for (int32 i=0;i<symtab.AvailableKey();i++) {
          ii=symtab.GetNthKey(i);
          std::string w=symtab.Find(ii);
          if (w=="</s>") {intlat2intrnn[ii]=0;intrnn2intlat[0]=ii;continue;}
          if (word2int_.find(w)!=word2int_.end())
            j=word2int_[w];
          else 
            j=--oov;

          intlat2intrnn[ii]=j; intrnn2intlat[j]=ii;
        } 
    }

    RNN() : ivcnt_(0),oovcnt_(0)  {
    }

    virtual ~RNN() {
    }

    BaseFloat Propagate(int32 lastW,int32 w,KaldiVector* hOut,const KaldiVector& hIn) {
      KaldiVector h_ac; // temporary variables for helping optimization
    //  FIXME REWRITE!!

/*
      h_ac.noalias()=-U1_*hIn; // h(t)=-U1*h(t-1)
      if (IsIV(lastW)) h_ac-=V1_.col(lastW); // h(t)=-h(t)-V1*w-1(t)

      // activate hidden layer (sigmoid) and determine updated s(t)
      *hOut=VectorXr(VectorXr(h_ac.array().exp()+1.0).array().inverse());

      if (IsIV(w)){
        // evaluate classes: c(t)=W*s(t) + activation class layer (softmax)
        cl_.noalias()=VectorXr((Cl_*(*hOut)).array().exp());
        // evaluate post. distribution for all words within that class: y(t)=V*s(t)
        int b=class2minint_[int2class_[w]];
        int n=class2maxint_[int2class_[w]]-b+1;

        // determine distribution of class of the predicted word
        // activate class part of the word layer (softmax)
        y_.segment(b,n).noalias()=VectorXr((W2_.middleRows(b,n)*(*hOut)).array().exp());
        ivcnt_++;
        return -log(y_(w)*cl_(int2class_[w])/cl_.sum()/y_.segment(b,n).sum());
      } else { oovcnt_++; return OOVPenalty(); }; */
      return 0; 
    }


  void TreeTraverse(CompactLattice* lat,const KaldiVector h) {
    // deal with the head of the tree explicitely, leave its scores untouched!
    for (fst::MutableArcIterator<CompactLattice> aiter(lat,lat->Start()); !aiter.Done(); aiter.Next()) { // follow <eps>
      for (fst::MutableArcIterator<CompactLattice> a2iter(lat,aiter.Value().nextstate); !a2iter.Done(); a2iter.Next()) { // follow <s>
        TreeTraverseRec(lat,a2iter.Value().nextstate,h,0); // call the visitor method with the given history and preceding <s> recursively
      }
    }
  }

  protected:

  void TreeTraverseRec(CompactLattice* lat, fst::StdFst::StateId i,const KaldiVector& lasth,int32 lastW) {
    KaldiVector h(HiddenSize());
    for (fst::MutableArcIterator<CompactLattice> aiter(lat, i); !aiter.Done(); aiter.Next()) {
      int32 w=intlat2intrnn[aiter.Value().olabel];  
      BaseFloat rnns=Propagate(lastW,w,&h,lasth);
      BaseFloat lms=aiter.Value().weight.Weight().Value1();
      BaseFloat ams=aiter.Value().weight.Weight().Value2();
      BaseFloat lmcs=rnns*Lambda()+lms*(1.0-Lambda()); // linear interpolation of log scores
      
      CompactLatticeArc newarc = aiter.Value();
      CompactLatticeWeight newwgt(LatticeWeight(lmcs,ams),aiter.Value().weight.String()); 
      newarc.weight=newwgt;
      aiter.SetValue(newarc);

      TreeTraverseRec(lat,aiter.Value().nextstate,h,w);
    }
  }
};


int main(int argc, char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Extracts N-best paths from lattices using given acoustic scale, \n"
        "rescores them using an RNN model and given lm scale and writes it out as FST.\n"
        "Usage: lattice-rnnrescore [options] dict lattice-rspecifier rnn-model lattice-wspecifier\n"
        " e.g.: lattice-rnnrescore --acoustic-scale=0.0625 --lambda=0.8 --oov-penalty=10 --n=10 WSJ.word-sym-tab ark:in.lats WSJ.rnn ark:nbest.lats\n";
      
    ParseOptions po(usage);
    BaseFloat lambda = 0.75;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat oov_penalty = 11; // assumes vocabularies around 60k words 
    RNN::int32 n = 10;
    

    po.Register("lambda", &lambda, "Weighting factor between 0 and 1 for the given RNN LM" );
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("oov-penalty", &oov_penalty, "A reasonable value is ln(vocab_size)" );
    po.Register("n", &n, "Number of distinct paths >= 1");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      KALDI_EXIT << "Wrong arguments!";
    }

    std::string wordsymtab_filename = po.GetArg(1),
        rnnmodel_filename = po.GetArg(3),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(4);

    // read the dictionary
    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(wordsymtab_filename)))
      KALDI_EXIT << "Could not read symbol table from file " << wordsymtab_filename;

    // read as regular lattice-- this is the form we need it in for efficient pruning.
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    RNN myRNN;
    bool binary;
    Input in(rnnmodel_filename,&binary);

    // load our RNN model
    myRNN.Read(in.Stream(),binary);
    in.Close();

    // FIXME DANS NOTE!
    // should probably be defining the RNN class in a .h and .cc file
    // let's call it RnnLm
    
    // give the word symbol table used in the input lattices
    myRNN.SetLatticeSymbols(*word_syms);

    RNN::int32 n_done = 0; // there is no failure mode, barring a crash.
    RNN::int64 n_paths_out = 0;

    if (acoustic_scale == 0.0)
      KALDI_EXIT << "Do not use a zero acoustic scale (cannot be inverted)";

    // set lambda/oov penalty
    myRNN.SetLambda(lambda); 
    myRNN.SetOOVPenalty(oov_penalty);

    KaldiVector h; 

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      Lattice rlat;

      lattice_reader.FreeCurrent();

      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);

      fst::Reverse(lat, &rlat);
      Lattice nbestr_lat, nbest_lat;
      fst::ShortestPath(rlat, &nbestr_lat, n);
      fst::Reverse(nbestr_lat, &nbest_lat);

      if (nbestr_lat.Start() != fst::kNoStateId)
        n_paths_out += nbestr_lat.NumArcs(nbestr_lat.Start());

      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &nbest_lat);

      CompactLattice nbest_clat;
      ConvertLattice(nbest_lat, &nbest_clat);

      // initialize the RNN
      h.Resize(myRNN.HiddenSize(),kSetZero); 
      myRNN.Propagate(0,0,&h,h);

      // recompute LM scores  
      myRNN.TreeTraverse(&nbest_clat,h);

      // write out the new lattice
      compact_lattice_writer.Write(key, nbest_clat);
      n_done++;
    }

    KALDI_LOG << "Did N-best algorithm to " << n_done << " lattices with n = "
              << n << ", average actual #paths is "
              << (n_paths_out/(n_done+1.0e-20));
    KALDI_LOG << "Done " << n_done << " utterances, " << myRNN.IVProcessed()+myRNN.OOVProcessed() << " RNN computations ("<<myRNN.OOVProcessed()<<" OOVs)!";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
