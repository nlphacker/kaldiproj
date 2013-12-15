// hmm/hmm-topology-test.cc

// Copyright 2009-2011 Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
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

#include "hmm/hmm-topology.h"

namespace kaldi {


void TestHmmTopology() {
  bool binary = (rand()%2 == 0);

  std::string input_str = "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>\n"
      "<State> 0 <PdfClass> 0\n"
      "<Transition> 0 0.5\n"
      "<Transition> 1 0.5\n"
      "</State> \n"
      "<State> 1 <PdfClass> 1 \n"
      "<Transition> 1 0.5\n"
      "<Transition> 2 0.5\n"
      "</State>  \n"
      " <State> 2 <PdfClass> 2\n"
      " <Transition> 2 0.5\n"
      " <Transition> 3 0.5\n"
      " </State>   \n"
      " <State> 3 </State>\n"
      " </TopologyEntry>\n"
      "  <TopologyEntry>\n"
      "  <ForPhones> 10 11 13  </ForPhones>\n"
      "  <State> 0 <PdfClass> 0\n"
      "  <Transition> 0 0.5\n"
      "  <Transition> 1 0.5\n"
      "  </State> \n"
      "  <State> 1 <PdfClass> 1 \n"
      "  <Transition> 1 0.5\n"
      "  <Transition> 2 0.5\n"
      "  </State>  \n"
      " <State> 2 </State>"
      "  </TopologyEntry>\n"
      "  </Topology>\n";

  HmmTopology topo;

  std::istringstream iss(input_str);
  topo.Read(iss, false);

  std::ostringstream oss;
  topo.Write(oss, binary);

  HmmTopology topo2;
  // std::cout << oss.str() << '\n' << std::flush;
  std::istringstream iss2(oss.str());
  topo2.Read(iss2, binary);

  {  // test equality.
    std::ostringstream oss1, oss2;
    topo.Write(oss1, false);
    topo2.Write(oss2, false);
    assert(oss1.str() == oss2.str());
  }
}


}

int main() {
  // repeat the test ten times
  for (int i = 0; i < 10; i++) {
    kaldi::TestHmmTopology();
  }
  std::cout << "Test OK.\n";
}

