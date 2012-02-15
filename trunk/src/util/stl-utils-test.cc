// util/stl-utils-test.cc

// Copyright 2009-2012     Microsoft Corporation;  Saarland University
//                         Daniel Povey

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
#include "util/stl-utils.h"

namespace kaldi {
static void TestIsSorted() {
  for (int i = 0;i < 100;i++) {
    std::vector<int> vec, vec2;
    int len = rand()%5;
    for (int i = 0;i < len;i++)
      vec.push_back(rand() % 10);
    vec2 = vec;
    std::sort(vec2.begin(), vec2.end());
    assert(IsSorted(vec) == (vec == vec2));
  }
}

static void TestIsSortedAndUniq() {
  for (int i = 0;i < 100;i++) {
    std::vector<int> vec, vec2;
    int len = rand()%5;
    for (int i = 0;i < len;i++)
      vec.push_back(rand() % 10);

    if (! IsSortedAndUniq(vec)) {
      bool ok = false;
      for (size_t i = 0; i+1 < (size_t)len; i++)
        if (vec[i] >= vec[i+1]) ok = true;  // found out-of-order or dup.
      assert(ok);
    } else {  // is sorted + uniq.
      for (size_t i = 0; i+1 < (size_t)len; i++)
        assert(vec[i] < vec[i+1]);
    }
  }
}


static void TestUniq() {
  for (int i = 0;i < 100;i++) {
    std::vector<int>  vec;

    int cur = 1;  //  sorted order.
    int len = rand()%5;
    for (int i = 0;i < len;i++) {
      cur += 1 + (rand() % 100);
      vec.push_back(cur);
    }
    std::vector<int> vec2;
    for (int i = 0;i < len;i++) {
      int count = 1 + rand()%5;
      for (int j = 0;j < count;j++) vec2.push_back(vec[i]);
    }
    Uniq(&vec2);
    assert(vec2 == vec);
  }
}


static void TestSortAndUniq() {
  for (int i = 0;i < 100;i++) {
    std::vector<int>  vec;

    int len = rand()%5;
    for (int i = 0;i < len;i++) {
      int n = rand() % 100;
      bool ok = true;
      for (size_t j = 0;j < vec.size();j++) if (vec[j] == n) ok = false;
      if (ok)  vec.push_back(n);
    }
    // don't sort.
    std::vector<int> vec2(vec);  // make sure all things in "vec" represented in vec2.
    int len2 = rand()%10;
    if (vec.size() > 0) // add more, randomly.
      for (int i = 0;i < len2;i++)
        vec2.push_back(vec[rand()%vec.size()]);
    SortAndUniq(&vec2);
    std::sort(vec.begin(), vec.end());
    assert(vec == vec2);
  }
}

void TestCopySetToVector() {
  for (int p = 0; p < 100; p++) {
    std::set<int> st;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) st.insert(rand() % 10);
    std::vector<int> v;
    CopySetToVector(st, &v);
    assert(st.size() == v.size());
    for (size_t i = 0;i < v.size();i++) assert(st.count(v[i]) != 0);
  }
}


void TestCopyMapToVector() {
  for (int p = 0; p < 100; p++) {
    std::map<int, int> mp;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) mp[rand() % 10] = rand() % 20;
    std::vector<std::pair<int, int> > v;
    CopyMapToVector(mp, &v);
    assert(mp.size() == v.size());
    for (size_t i = 0;i < v.size();i++) assert(mp[v[i].first] == v[i].second);
  }
}


void TestCopyMapKeysToVector() {
  for (int p = 0; p < 100; p++) {
    std::map<int, int> mp;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) mp[rand() % 10] = rand() % 20;
    std::vector<int> v;
    CopyMapKeysToVector(mp, &v);
    assert(mp.size() == v.size());
    for (size_t i = 0;i < v.size();i++) assert(mp.count(v[i]) == 1);
  }
}

void TestCopyMapValuesToVector() {
  for (int p = 0; p < 100; p++) {
    std::map<int, int> mp;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) mp[rand() % 10] = rand() % 20;
    std::vector<int> v;
    CopyMapValuesToVector(mp, &v);
    assert(mp.size() == v.size());
    int i = 0;
    for (std::map<int, int>::iterator iter = mp.begin(); iter != mp.end(); iter++) {
      assert(v[i++] == iter->second);
    }
  }
}

void TestCopyMapKeysToSet() {
  for (int p = 0; p < 100; p++) {
    std::map<int, int> mp;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) mp[rand() % 10] = rand() % 20;
    std::vector<int> v;
    std::set<int> s;
    CopyMapKeysToVector(mp, &v);
    CopyMapKeysToSet(mp, &s);
    std::set<int> s2;
    CopyVectorToSet(v, &s2);
    assert(s == s2);
  }
}


void TestCopyMapValuesToSet() {
  for (int p = 0; p < 100; p++) {
    std::map<int, int> mp;
    int sz = rand() % 20;
    for (int i = 0;i < sz;i++) mp[rand() % 10] = rand() % 20;
    std::vector<int> v;
    std::set<int> s;
    CopyMapValuesToVector(mp, &v);
    CopyMapValuesToSet(mp, &s);
    std::set<int> s2;
    CopyVectorToSet(v, &s2);
    assert(s == s2);
  }
}


void TestContainsNullPointers() {
  for (int p = 0; p < 100; p++) {
    std::vector<char*> vec;
    int sz = rand() % 3;
    bool is_null = false;
    for (int i = 0;i < sz;i++) { vec.push_back( (char*) (rand() % 2)); if (vec.back() == NULL) is_null = true; }
    assert(is_null == ContainsNullPointers(vec));
  }
}

void TestReverseVector() {
  for (int p = 0; p < 100; p++) {
    std::vector<int> vec;
    int sz = rand() % 5;
    for (int i = 0;i < sz;i++)
      vec.push_back( rand() % 4) ;
    std::vector<int> vec2(vec), vec3(vec);
    ReverseVector(&vec2);
    ReverseVector(&vec2);
    assert(vec2 == vec);
    ReverseVector(&vec3);
    for (size_t i = 0; i < vec.size(); i++)
      assert(vec[i] == vec3[vec.size()-1-i]);
  }
}

void TestMergePairVectorSumming() {
  for (int p = 0; p < 100; p++) {
    std::vector<std::pair<int32, int16> > v;
    std::map<int32, int16> m;
    int sz = rand() % 10;
    for (size_t i = 0; i < sz; i++) {
      int32 key = rand() % 10;
      int16 val = (rand() % 5) - 2;
      v.push_back(std::make_pair(key, val));
      if (m.count(key) == 0) m[key] = val;
      else m[key] += val;
    }
    MergePairVectorSumming(&v);
    KALDI_ASSERT(IsSorted(v));
    for (size_t i = 0; i < v.size(); i++) {
      KALDI_ASSERT(v[i].second == m[v[i].first]);
      KALDI_ASSERT(v[i].second != 0.0);
      if (i > 0) KALDI_ASSERT(v[i].first > v[i-1].first);
    }
    for (std::map<int32, int16>::const_iterator iter = m.begin();
         iter != m.end(); ++iter) {
      if (iter->second != 0) {
        size_t i;
        for (i = 0; i < v.size(); i++)
          if (v[i].first == iter->first) break;
        KALDI_ASSERT(i != v.size()); // Or we didn't find this
        // key in v.
      }
    }
  }
}
  

} // end namespace kaldi

int main() {
  using namespace kaldi;
  TestIsSorted();
  TestIsSortedAndUniq();
  TestUniq();
  TestSortAndUniq();
  TestCopySetToVector();
  TestCopyMapToVector();
  TestCopyMapKeysToVector();
  TestCopyMapValuesToVector();
  TestCopyMapKeysToSet();
  TestCopyMapValuesToSet();
  TestContainsNullPointers();
  TestReverseVector();
  TestMergePairVectorSumming();
  // CopyVectorToSet implicitly tested by last 2.
  std::cout << "Test OK\n";
}



