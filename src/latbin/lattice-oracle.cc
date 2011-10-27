// latbin/lattice-oracle.cc

// Copyright 2011 Gilles Boulianne
//
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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

using std::vector;
using std::set;

typedef vector<fst::StdArc::Label> LabelVector;
typedef set<fst::StdArc::Label> LabelSet; 

LabelVector *GetSymbolSet(const fst::StdVectorFst &fst, bool inputSide) {
  LabelVector *lv = new LabelVector();
  for (fst::StateIterator<fst::StdVectorFst> siter(fst); !siter.Done(); siter.Next()) {
    fst::StdArc::StateId s = siter.Value();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      lv->push_back(inputSide ? arc.ilabel : arc.olabel);
    }
  }  
  return lv;
}

void ReadSymbolList(const std::string &filename,
                    fst::SymbolTable *word_syms,
                    LabelSet *lset) {
  std::ifstream is(filename.c_str());
  if (!is.good()) 
    KALDI_ERR << "ReadSymbolList: could not open symbol list "<<filename;
  std::string line;
  assert(lset != NULL);
  lset->clear();
  while (getline(is, line)) {
    std::string sym;
    std::istringstream ss(line);
    ss >> sym >> std::ws;
    if (ss.fail() || !ss.eof()) {
      KALDI_ERR << "Bad line in symbol list: "<< line<<", file is: "<<filename;
    }
    fst::StdArc::Label lab = word_syms->Find(sym.c_str());
    if (lab == fst::SymbolTable::kNoSymbol) {
      KALDI_ERR << "Can't find symbol in symbol table: "<< line<<", file is: "<<filename;
    }
    cerr << "ReadSymbolList: adding symbol "<<sym<<" ("<<lab<<")"<<endl;
    lset->insert(lab);
  }
}

void MapWildCards(const LabelSet &wildcards, fst::StdVectorFst *ofst) {
  // map all wildcards symbols to epsilons
  for (fst::StateIterator<fst::StdVectorFst> siter(*ofst); !siter.Done(); siter.Next()) {
    fst::StdArc::StateId s = siter.Value();
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(ofst, s); !aiter.Done();  aiter.Next()) {
      fst::StdArc arc(aiter.Value());
      LabelSet::iterator it = wildcards.find(arc.ilabel);
      if (it != wildcards.end()) {
        cerr << "MapWildCards: mapping symbol "<<arc.ilabel<<" to epsilon"<<endl;
        arc.ilabel = 0;
      }
      it = wildcards.find(arc.olabel);
      if (it != wildcards.end()) {arc.olabel = 0;}
      aiter.SetValue(arc);
    }
  }    
}

// convert from Lattice to standard FST
// also maps wildcard symbols to epsilons
// then removes epsilons
void ConvertLatticeToUnweightedAcceptor(const kaldi::Lattice& ilat,
                                        const LabelSet &wildcards,
                                        fst::StdVectorFst *ofst) {
  // first convert from  lattice to normal FST
  fst::ConvertLattice(ilat, ofst); 
  // remove weights, project to output, sort according to input arg
  fst::Map(ofst, fst::RmWeightMapper<fst::StdArc>()); 
  fst::Project(ofst, fst::PROJECT_OUTPUT);  // The words are on the output side  
  MapWildCards(wildcards,ofst);
  fst::RmEpsilon(ofst);   // Don't tolerate epsilons as they make it hard to tally errors
  fst::ArcSort(ofst, fst::StdILabelCompare());
}

void CreateEditDistance(const fst::StdVectorFst &fst1,
                        const fst::StdVectorFst &fst2,
                        fst::StdVectorFst *pfst) {
  fst::StdArc::Weight corrCost(0.0);
  fst::StdArc::Weight subsCost(1.0);
  fst::StdArc::Weight insCost(1.0);
  fst::StdArc::Weight delCost(1.0);

  // create set of output symbols in fst1
  LabelVector *fst1syms = GetSymbolSet(fst1,false);
  
  // create set of input symbols in fst2
  LabelVector *fst2syms = GetSymbolSet(fst2,true);

  pfst->AddState();
  pfst->SetStart(0);
  for (LabelVector::iterator it=fst1syms->begin(); it<fst1syms->end(); it++) {
    pfst->AddArc(0,fst::StdArc(*it,0,delCost,0));    // deletions
  }
  for (LabelVector::iterator it=fst2syms->begin(); it<fst2syms->end(); it++) {
    pfst->AddArc(0,fst::StdArc(0,*it,insCost,0));    // insertions
  }
  // stupid implementation O(N^2)
  for (LabelVector::iterator it1=fst1syms->begin(); it1<fst1syms->end(); it1++) {
    for (LabelVector::iterator it2=fst2syms->begin(); it2<fst2syms->end(); it2++) {
      fst::StdArc::Weight cost( (*it1) == (*it2) ? corrCost : subsCost);
      pfst->AddArc(0,fst::StdArc((*it1),(*it2),cost,0));    // substitutions
    }
  }
  pfst->SetFinal(0,fst::StdArc::Weight::One());
  fst::ArcSort(pfst, fst::StdOLabelCompare());
}

void CountErrors(fst::StdVectorFst &fst,
                 unsigned int *corr,
                 unsigned int *subs,
                 unsigned int *ins,
                 unsigned int *del,
                 unsigned int *totwords) {
 
   *corr = *subs = *ins = *del = *totwords = 0;

  // go through the first complete path in fst (there should be only one)
  fst::StdArc::StateId src = fst.Start(); 
  while (fst.Final(src)== fst::StdArc::Weight::Zero()) { // while not final
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst,src); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      if (arc.ilabel == 0 && arc.olabel == 0) {
        // don't count these so we may compare number of arcs and number of errors
      } else if (arc.ilabel == arc.olabel) {
        (*corr)++; (*totwords)++;
      } else if (arc.ilabel == 0) {
        (*ins)++;
      } else if (arc.olabel == 0) {
        (*del)++; (*totwords)++;
      } else {
        (*subs)++; (*totwords)++;
      }
      src = arc.nextstate;
      continue; // jump to next state
    }
  }
}


bool CheckFst(fst::StdVectorFst &fst, string name, string key) {

#ifdef DEBUG
  fst::StdArc::StateId numstates = fst.NumStates();
  cerr << " "<<name<<" has "<<numstates<<" states"<<endl;
  std::stringstream ss; ss <<name<<key<<".fst";
  fst.Write(ss.str());
  return(fst.Start() == fst::kNoStateId); 
#else
  return true;
#endif
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Finds the path having the smallest edit-distance between two lattices.\n"
        "For efficiency put the smallest lattices first (for example reference strings).\n"
        "Usage: lattice-oracle [options] test-lattice-rspecifier reference-rspecifier transcriptions-wspecifier\n"
        " e.g.: lattice-oracle ark:ref.lats ark:1.lats ark:1.tra\n";
        
    ParseOptions po(usage);
    
    std::string word_syms_filename;
    std::string wild_syms_filename;

    std::string lats_wspecifier;
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("wildcard-symbols-list", &wild_syms_filename, "List of symbols that don't count as errors");
    po.Register("write-lattices", &lats_wspecifier, "If supplied, write 1-best path as lattices to this wspecifier");
    
    po.Read(argc, argv);
 
    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        reference_rspecifier = po.GetArg(2),
        transcriptions_wspecifier = po.GetOptArg(3);

    // will read input as  lattices
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    RandomAccessInt32VectorReader reference_reader(reference_rspecifier);

    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
        << word_syms_filename;

    LabelSet wild_syms;
    if (wild_syms_filename != "") 
      ReadSymbolList(wild_syms_filename, word_syms, &wild_syms);
    
    int32 n_done = 0, n_fail = 0;
    unsigned int tot_corr=0, tot_subs=0, tot_ins=0, tot_del=0, tot_words=0;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      const Lattice &lat = lattice_reader.Value();
      cerr << "Lattice "<<key<<" read."<<endl;

      // remove all weights while creating a standard FST
      VectorFst<StdArc> fst1;
      ConvertLatticeToUnweightedAcceptor(lat,wild_syms,&fst1);
      CheckFst(fst1, "fst1_", key);
      
      // TODO: map certain symbols (using an FST created with CreateMapFst())
      
      if (!reference_reader.HasKey(key)) {
        KALDI_WARN << "No reference present for utterance " << key;
        n_fail++;
        continue;
      }
      const std::vector<int32> &reference = reference_reader.Value(key);
      VectorFst<StdArc> fst2;
      MakeLinearAcceptor(reference, &fst2);
      
      CheckFst(fst2, "fst2_", key);
            
      // recreate edit distance fst if necessary
      fst::StdVectorFst editDistanceFst;
      CreateEditDistance(fst1, fst2, &editDistanceFst);
      
      // compose with edit distance transducer
      VectorFst<StdArc> composedFst;
      fst::Compose(editDistanceFst, fst2, &composedFst);
      CheckFst(composedFst, "composed_", key);
      
      // make sure composed FST is input sorted
      fst::ArcSort(&composedFst, fst::StdILabelCompare());
      
      // compose with previous result
      VectorFst<StdArc> resultFst;
      fst::Compose(fst1, composedFst, &resultFst);
      CheckFst(resultFst, "result_", key);
      
      // find out best path
      VectorFst<StdArc> best_path;
      fst::ShortestPath(resultFst, &best_path);
      CheckFst(best_path, "best_path_", key);

      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_fail++;
      } else {

        // count errors
        unsigned int corr, subs, ins, del, totwords;
        CountErrors(best_path, &corr, &subs, &ins, &del, &totwords);
        unsigned int toterrs = subs+ins+del;
        KALDI_LOG << "%WER "<<(100.*toterrs)/totwords<<" [ "<<toterrs<<" / "<<totwords<<", "<<ins<<" ins, "<<del<<" del, "<<subs<<" sub ]";
        tot_corr += corr; tot_subs += subs; tot_ins += ins; tot_del += del; tot_words += totwords;     
        
        std::vector<int32> oracle_words;
        std::vector<int32> reference_words;
        fst::StdArc::Weight weight;
        GetLinearSymbolSequence(best_path, &oracle_words, &reference_words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost " << weight;
        if (transcriptions_wspecifier != "")
          transcriptions_writer.Write(key, oracle_words);
        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < oracle_words.size(); i++) {
            std::string s = word_syms->Find(oracle_words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << oracle_words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
      }
      n_done++;
    }
    if (word_syms) delete word_syms;
    unsigned int tot_errs = tot_subs + tot_del + tot_ins;
    KALDI_LOG << "%WER "<<(100.*tot_errs)/tot_words<<" [ "<<tot_errs<<" / "<<tot_words<<", "<<tot_ins<<" ins, "<<tot_del<<" del, "<<tot_subs<<" sub ]";
    KALDI_LOG << "Scored " << n_done << " lattices, "<<n_fail<<" not present in hyp.";
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
