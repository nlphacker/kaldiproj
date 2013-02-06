// latbin/lattice-1best.cc

// Copyright 2012  Johns Hopkins University (Author: Ehsan Variani)

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
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    using fst::VectorFst;
    using fst::StdArc;
    typedef StdArc::StateId StateId;
    
    const char *usage = 
      "Usage: lattice-depth <lattice-rspecifier> [<depth-wspecifier>]";

    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    std::string depth_wspecifier = po.GetArg(2);
    BaseFloatWriter lats_depth_writer(depth_wspecifier);

    int64 num_done = 0, sum_depth = 0, total_num_utt = 0;
    for (; clat_reader.Done(); clat_reader.Next()) {
      int64 depth = 0;
      const CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();
      CompactLattice copy_lat = clat;
      Lattice lat;
      ConvertLattice(copy_lat, &lat);
      vector<int32> state_times;
      int32 num_utt = kaldi::LatticeStateTimes(lat, &state_times);
      total_num_utt += num_utt;
      for (StateId s = 0; s < clat.NumStates(); s++) {
	for (fst::ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done();
	     aiter.Next()) {
	  const CompactLatticeArc &arc = aiter.Value();
	  depth += arc.weight.String().size();
	}
      }
      if (depth_wspecifier != "") {
	lats_depth_writer.Write(key, static_cast<float> ((float)depth / num_utt));
      }
      sum_depth += depth;
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << "lattices.";
    KALDI_LOG << "The average density is "
	      << static_cast<float> ((float)sum_depth / total_num_utt);
    if (num_done != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
