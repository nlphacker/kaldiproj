// base/io-funcs-test.cc

// Copyright 2009-2011  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "base/io-funcs.h"
#include "base/kaldi-math.h"

namespace kaldi {

void UnitTestIo(bool binary) {
  {
    const char *filename = "tmpf";
    std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
    InitKaldiOutputStream(outfile, binary);
    if (!binary) outfile << "\t";
    int64 i1 = rand() % 10000;
    WriteBasicType(outfile, binary, i1);
    uint16 i2 = rand() % 10000;
    WriteBasicType(outfile, binary, i2);
    if (!binary) outfile << "\t";
    char c = rand();
    WriteBasicType(outfile, binary, c);
    if (!binary && rand()%2 == 0) outfile << " \n";
    std::vector<int32> vec1;
    WriteIntegerVector(outfile, binary, vec1);
    if (!binary && rand()%2 == 0) outfile << " \n";
    std::vector<uint16> vec2;
    for (size_t i = 0; i < 10; i++) vec2.push_back(rand()%100 - 10);
    WriteIntegerVector(outfile, binary, vec2);
    if (!binary) outfile << " \n";
    std::vector<char> vec3;
    for (size_t i = 0; i < 10; i++) vec3.push_back(rand()%100);
    WriteIntegerVector(outfile, binary, vec3);
    if (!binary && rand()%2 == 0) outfile << " \n";
    const char *marker1 = "Hi";
    WriteMarker(outfile, binary, marker1);
    if (!binary) outfile << " \n";
    std::string marker2 = "There.";
    WriteMarker(outfile, binary, marker2);
    if (!binary && rand()%2 == 0) outfile << " \n";
    std::string marker3 = "You.";
    WriteMarker(outfile, binary, marker3);
    if (!binary && rand()%2 == 0) outfile << " ";
    float f1 = RandUniform();
    WriteBasicType(outfile, binary, f1);
    if (!binary && rand()%2 == 0) outfile << "\t";
    float f2 = RandUniform();
    WriteBasicType(outfile, binary, f2);
    double d1 = RandUniform();
    WriteBasicType(outfile, binary, d1);
    if (!binary && rand()%2 == 0) outfile << "\t";
    double d2 = RandUniform();
    WriteBasicType(outfile, binary, d2);
    if (!binary && rand()%2 == 0) outfile << "\t";
    outfile.close();

    {
      std::ifstream infile(filename, std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(infile, &binary_in);
      int64 i1_in;
      ReadBasicType(infile, binary_in, &i1_in);
      assert(i1_in == i1);
      uint16 i2_in;
      ReadBasicType(infile, binary_in, &i2_in);
      assert(i2_in == i2);
      char c_in;
      ReadBasicType(infile, binary_in, &c_in);
      assert(c_in == c);
      std::vector<int32> vec1_in;
      ReadIntegerVector(infile, binary_in, &vec1_in);
      assert(vec1_in == vec1);
      std::vector<uint16> vec2_in;
      ReadIntegerVector(infile, binary_in, &vec2_in);
      assert(vec2_in == vec2);
      std::vector<char> vec3_in;
      ReadIntegerVector(infile, binary_in, &vec3_in);
      assert(vec3_in == vec3);
      std::string  marker1_in, marker2_in;
      assert(PeekMarker(infile, binary_in) == (int)*marker1);
      ReadMarker(infile, binary_in, &marker1_in);
      assert(marker1_in == (std::string)marker1);
      ReadMarker(infile, binary_in, &marker2_in);
      assert(marker2_in == marker2);
      if (rand() % 2 == 0)
        ExpectMarker(infile, binary_in, marker3.c_str());
      else
        ExpectMarker(infile, binary_in, marker3);
      float f1_in;  // same type.
      ReadBasicType(infile, binary_in, &f1_in);
      AssertEqual(f1_in, f1);
      double f2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &f2_in);
      AssertEqual(f2_in, f2);
      double d1_in;  // same type.
      ReadBasicType(infile, binary_in, &d1_in);
      AssertEqual(d1_in, d1);
      float d2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &d2_in);
      AssertEqual(d2_in, d2);
      assert(PeekMarker(infile, binary_in) == -1);
    }
  }
}



}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  for (size_t i = 0; i < 10; i++) {
    UnitTestIo(false);
    UnitTestIo(true);
  }
  KALDI_ASSERT(1);  // just wanted to check that KALDI_ASSERT does not fail for 1.
  return 0;
}

