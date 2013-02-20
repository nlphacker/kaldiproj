// tree/build-tree.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include <set>
#include <queue>
#include "util/stl-utils.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {


void GenRandStats(int32 dim, int32 num_stats, int32 N, int32 P,
                  const std::vector<int32> &phone_ids,
                  const std::vector<int32> &phone2hmm_length,
                  const std::vector<bool> &is_ctx_dep,
                  bool ensure_all_phones_covered,
                  BuildTreeStatsType *stats_out) {

  assert(dim > 0);
  assert(num_stats > 0);
  assert(N > 0);
  assert(P < N);
  assert(phone_ids.size() != 0);
  assert(stats_out != NULL && stats_out->empty());
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());
  assert(phone2hmm_length.size() >= static_cast<size_t>(1 + max_phone));
  assert(is_ctx_dep.size() >= static_cast<size_t>(1 + max_phone));

  // Make sure phone id's distinct.
  {
    std::vector<int32> tmp(phone_ids);
    SortAndUniq(&tmp);
    assert(tmp.size() == phone_ids.size());
  }
  size_t num_phones = phone_ids.size();

  // Decide on an underlying "mean" for phones...
  Matrix<BaseFloat> phone_vecs(max_phone+1, dim);
  for (int32 i = 0;i < max_phone+1;i++)
    for (int32 j = 0;j < dim;j++) phone_vecs(i, j) = RandGauss() * (2.0 / (j+1));


  std::map<EventType, Clusterable*> stats_tmp;

  std::vector<bool> covered(1 + max_phone, false);

  bool all_covered = false;
  for (int32 i = 0;i < num_stats || (ensure_all_phones_covered && !all_covered);i++) {
    // decide randomly on a phone-in-context.
    std::vector<int32> phone_vec(N);
    for (size_t i = 0;i < (size_t)N;i++) phone_vec[i] = phone_ids[(rand() % num_phones)];

    int32 hmm_length = phone2hmm_length[phone_vec[P]];
    KALDI_ASSERT(hmm_length > 0);
    covered[phone_vec[P]] = true;

    // For each position [in the central phone]...
    for (int32 j = 0; j < hmm_length; j++) {
      // create event vector.
      EventType event_vec;
      event_vec.push_back(std::make_pair(kPdfClass, (EventValueType)j));  // record the position.
      for (size_t pos = 0; pos < (size_t)N; pos++) {
        if (pos == (size_t)(P) || is_ctx_dep[phone_vec[P]])
          event_vec.push_back(std::make_pair((EventKeyType)pos, (EventValueType)phone_vec[pos]));
        // The if-statement above ensures we do not record the context of "context-free"
        // phone (e.g., silence).
      }

      Vector<BaseFloat> mean(dim);  // mean of Gaussian.
      GaussClusterable *this_stats = new GaussClusterable(dim, 0.1);  // 0.1 is var floor.
      {  // compute stats; this block attempts to simulate the process of "real" data
        // collection and does not correspond to any code you would write in a real
        // scenario.
        Vector<BaseFloat> weights(N);  // weight of each component.
        for (int32 k = 0; k < N; k++) {
          BaseFloat k_pos = (N - 0.5 - k) / N;  // between 0 and 1, less for lower k...
          BaseFloat j_pos = (hmm_length - 0.5 - j) / hmm_length;
          // j_pos is between 0 and 1, less for lower j.

          BaseFloat weight = j_pos*k_pos + (1.0-j_pos)*(1.0-k_pos);
          // if j_pos close to zero, gives larger weight to k_pos close
          // to zero.
          if (k == P) weight += 1.0;
          weights(k) = weight;
        }
        assert(weights.Sum() != 0);
        weights.Scale(1.0 / weights.Sum());
        for (int32 k = 0; k < N; k++)
          mean.AddVec(weights(k), phone_vecs.Row(phone_vec[k]));
        BaseFloat count;
        if (rand() % 2 == 0) count = 1000.0 * RandUniform();
        else count = 100.0 * RandUniform();

        int32 num_samples = 10;
        for (size_t p = 0;p < (size_t)num_samples; p++) {
          Vector<BaseFloat> sample(mean);  // copy mean.
          for (size_t d = 0; d < (size_t)dim; d++)  sample(d) += RandGauss();  // unit var.
          this_stats->AddStats(sample, count / num_samples);
        }
      }

      if (stats_tmp.count(event_vec) != 0) {
        stats_tmp[event_vec]->Add(*this_stats);
        delete this_stats;
      } else {
        stats_tmp[event_vec] = this_stats;
      }
    }
    all_covered = true;
    for (size_t i = 0; i< num_phones; i++) if (!covered[phone_ids[i]]) all_covered = false;
  }
  CopyMapToVector(stats_tmp, stats_out);
  KALDI_ASSERT(stats_out->size() > 0);
}



EventMap *BuildTree(Questions &qopts,
                    const std::vector<std::vector<int32> > &phone_sets,
                    const std::vector<int32> &phone2num_pdf_classes,
                    const std::vector<bool> &share_roots,
                    const std::vector<bool> &do_split,
                    const BuildTreeStatsType &stats,
                    BaseFloat thresh,
                    int32 max_leaves,
                    BaseFloat cluster_thresh,  // typically == thresh.  If negative, use smallest split.
                    int32 P) {
  assert(thresh > 0 || max_leaves > 0);
  assert(stats.size() != 0);
  assert(!phone_sets.empty()
         && phone_sets.size() == share_roots.size()
         && do_split.size() == phone_sets.size());

  for (size_t i = 0; i < do_split.size(); i++)
    if (do_split[i])
      KALDI_ASSERT(share_roots[i] && "We don't support splitting non-shared roots.");

  // the inputs will be further checked in GetStubMap.
  int32 num_leaves = 0;  // allocator for leaves.

  EventMap *tree_stub = GetStubMap(P,
                                   phone_sets,
                                   phone2num_pdf_classes,
                                   share_roots,
                                   &num_leaves);
  KALDI_VLOG(1) <<  "BuildTree: before building trees, map has "<< num_leaves << " leaves.";


  BaseFloat impr;
  BaseFloat smallest_split = 1.0e+10;


  std::vector<int32> nonsplit_phones;
  for (size_t i = 0; i < phone_sets.size(); i++)
    if (!do_split[i])
      nonsplit_phones.insert(nonsplit_phones.end(), phone_sets[i].begin(), phone_sets[i].end());

  std::sort(nonsplit_phones.begin(), nonsplit_phones.end());

  assert(IsSortedAndUniq(nonsplit_phones));
  BuildTreeStatsType filtered_stats;
  if (!nonsplit_phones.empty())
    FilterStatsByKey(stats, P, nonsplit_phones, false,  // retain only those not
                     // in "nonsplit_phones"
                     &filtered_stats);

  EventMap *tree_split = SplitDecisionTree(*tree_stub,
                                           nonsplit_phones.empty() ? stats : filtered_stats,
                                           qopts, thresh, max_leaves,
                                           &num_leaves, &impr, &smallest_split);

  if (cluster_thresh < 0.0) {
    KALDI_VLOG(1) <<  "Setting clustering threshold to smallest split " << smallest_split;
    cluster_thresh = smallest_split;
  }

  BaseFloat normalizer = SumNormalizer(stats), impr_normalized = impr / normalizer;

  KALDI_VLOG(1) <<  "After decision tree split, num-leaves = "<< num_leaves
                << ", like-impr = " << impr_normalized << " per frame over "
                << normalizer << " frames.";

  if (cluster_thresh != 0.0) {   // Cluster the tree.
    BaseFloat objf_before_cluster = ObjfGivenMap(stats, *tree_split);

    // Now do the clustering.
    int32 num_removed = 0;
    EventMap *tree_clustered = ClusterEventMapRestrictedByMap(*tree_split,
                                                              stats,
                                                              cluster_thresh,
                                                              *tree_stub,
                                                              &num_removed);
    KALDI_VLOG(1) <<  "BuildTree: removed "<< num_removed << " leaves.";

    int32 num_leaves = 0;
    EventMap *tree_renumbered = RenumberEventMap(*tree_clustered, &num_leaves);

    BaseFloat objf_after_cluster = ObjfGivenMap(stats, *tree_renumbered);

    KALDI_VLOG(1) << "BuildTree: objf change due to clustering "
                  << ((objf_after_cluster-objf_before_cluster) / SumNormalizer(stats))
                  << " per frame.";
    KALDI_VLOG(1) <<  "BuildTreeClustered: num-leaves now "<< num_leaves;
    delete tree_clustered;
    delete tree_split;
    delete tree_stub;
    return tree_renumbered;
  } else {
    delete tree_stub;
    return tree_split;
  }
}


void ReadSymbolTableAsIntegers(std::string filename,
                               bool include_eps,
                               std::vector<int32> *syms) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    KALDI_ERR << "ReadSymbolTableAsIntegers: could not open symbol table "<<filename;
  std::string line;
  assert(syms != NULL);
  syms->clear();
  while (getline(is, line)) {
    std::string sym;
    int64 index;
    std::istringstream ss(line);
    ss >> sym >> index >> std::ws;
    if (ss.fail() || !ss.eof()) {
      KALDI_ERR << "Bad line in symbol table: "<< line<<", file is: "<<filename;
    }
    if (include_eps || index != 0)
      syms->push_back(index);
    if (index == 0 && sym != "<eps>") {
      KALDI_WARN << "Symbol zero is "<<sym<<", traditionally <eps> is used.  Make sure this is not a \"real\" symbol.";
    }
  }
  size_t sz = syms->size();
  SortAndUniq(syms);
  if (syms->size() != sz)
    KALDI_ERR << "Symbol table "<<filename<<" seems to contain duplicate symbols.";
}


/// ObtainSetsOfPhones is called by AutomaticallyObtainQuestions.
/// It processes the output of ClusterTopDown to obtain the sets
/// of phones corresponding to both the leaf-level clusters
/// and all the non-leaf-level clusters.
static void ObtainSetsOfPhones(const std::vector<std::vector<int32> > &phone_sets,  // the original phone sets, may
                               // just be individual phones.
                               const std::vector<int32> &assignments,  // phones->clusters
                               const std::vector<int32> &clust_assignments,  // clust->parent
                               int32 num_leaves,  // number of clusters present..
                               std::vector<std::vector<int32> > *sets_out) {
  assert(sets_out != NULL);
  sets_out->clear();
  std::vector<std::vector<int32> > raw_sets(clust_assignments.size());

  assert(num_leaves < static_cast<int32>(clust_assignments.size()));
  assert(assignments.size() == phone_sets.size());
  for (size_t i = 0; i < assignments.size(); i++) {
    int32 clust = assignments[i];  // this is an index into phone_sets.
    assert(clust>=0 && clust < num_leaves);
    for (size_t j = 0; j < phone_sets[i].size(); j++) {
       // and not just a hole.
      raw_sets[clust].push_back(phone_sets[i][j]);
    }
  }
  // for all clusters including the top-level cluster:
  // [note that the top-level cluster contains all phones, but it may actually
  //  be useful because sometimes we cluster just the non-silence phones, so
  //  the list of all phones is a way of asking about silence in such a way
  // that epsilon (end-or-begin-of-utterance) gets lumped with silence.
  for (int32 j = 0; j < static_cast<int32>(clust_assignments.size()); j++) {
    int32 parent = clust_assignments[j];
    std::sort(raw_sets[j].begin(), raw_sets[j].end());
    assert(IsSortedAndUniq(raw_sets[j]));  // should be no dups.
    if (parent < static_cast<int32>(clust_assignments.size())-1) {  // parent is not out of range [i.e. not the top one]...
      // add all j's phones to its parent.
      raw_sets[parent].insert(raw_sets[parent].end(),
                              raw_sets[j].begin(),
                              raw_sets[j].end());
    }
  }
  // Now add the original sets-of-phones to the raw sets, to make sure all of
  // these are present.  (The main reason they might be absent is if the stats
  // are empty, but we want to ensure they are all there regardless).  note these
  // will be actual singleton sets if the sets-of-phones each contain just one
  // phone, which in some sense is the normal situation.
  for (size_t i = 0; i < phone_sets.size(); i++) {
    raw_sets.push_back(phone_sets[i]);
  }
  // Remove duplicate sets from "raw_sets".
  SortAndUniq(&raw_sets);
  sets_out->reserve(raw_sets.size());
  for (size_t i = 0; i < raw_sets.size(); i++) {
    if (! raw_sets[i].empty()) // if the empty set is present, remove it...
      sets_out->push_back(raw_sets[i]);
  }
}


void AutomaticallyObtainQuestions(BuildTreeStatsType &stats,
                                  const std::vector<std::vector<int32> > &phone_sets_in,
                                  const std::vector<int32> &all_hmm_positions_in,
                                  int32 P,
                                  std::vector<std::vector<int32> > *questions_out) {
  std::vector<std::vector<int32> > phone_sets(phone_sets_in);
  std::vector<int32> phones;
  for (size_t i = 0; i < phone_sets.size() ;i++) {
    std::sort(phone_sets[i].begin(), phone_sets[i].end());
    if (phone_sets[i].empty())
      KALDI_ERR << "Empty phone set in AutomaticallyObtainQuestions";
    if (!IsSortedAndUniq(phone_sets[i]))
      KALDI_ERR << "Phone set in AutomaticallyObtainQuestions contains duplicate phones";
    for (size_t j = 0; j < phone_sets[i].size(); j++)
      phones.push_back(phone_sets[i][j]);
  }
  std::sort(phones.begin(), phones.end());
  if (!IsSortedAndUniq(phones))
    KALDI_ERR << "Phones are present in more than one phone set.";
  if (phones.empty())
    KALDI_ERR << "No phones provided.";

  std::vector<int32> all_hmm_positions(all_hmm_positions_in);
  SortAndUniq(&all_hmm_positions);
  assert(!all_hmm_positions.empty());

  BuildTreeStatsType retained_stats;
  FilterStatsByKey(stats, kPdfClass, all_hmm_positions,
                   true,  // retain only the listed positions
                   &retained_stats);


  std::vector<BuildTreeStatsType> split_stats;  // split by phone.
  SplitStatsByKey(retained_stats, P, &split_stats);

  std::vector<Clusterable*> summed_stats;  // summed up by phone.
  SumStatsVec(split_stats, &summed_stats);

  int32 max_phone = phones.back();
  if (static_cast<int32>(summed_stats.size()) < max_phone+1) {
    // this can happen if the last phone had no data.. if we are using
    // stress-marked, position-marked phones, this can happen.  The later
    // code will assume that a summed_stats entry exists for all phones.
    summed_stats.resize(max_phone+1, NULL);
  }

  for (int32 i = 0; static_cast<size_t>(i) < summed_stats.size(); i++) {  // A check.
    if (summed_stats[i] != NULL &&
       !binary_search(phones.begin(), phones.end(), i)) {
      KALDI_WARN << "Phone "<< i << " is present in stats but is not in phone list [make sure you intended this].";
    }
  }

  EnsureClusterableVectorNotNull(&summed_stats);  // make sure no NULL pointers in summed_stats.
  // will replace them with pointers to empty stats.

  std::vector<Clusterable*> summed_stats_per_set(phone_sets.size(), NULL);  // summed up by set.
  for (size_t i = 0; i < phone_sets.size(); i++) {
    const std::vector<int32> &this_set = phone_sets[i];
    summed_stats_per_set[i] = summed_stats[this_set[0]]->Copy();
    for (size_t j = 1; j < this_set.size(); j++)
      summed_stats_per_set[i]->Add(*(summed_stats[this_set[j]]));
  }

  for (size_t i = 0; i < summed_stats_per_set.size(); i++) {  // A check.
    if (summed_stats_per_set[i]->Normalizer() == 0.0) {
      std::ostringstream ss;
      ss << "AutomaticallyObtainQuestions: no stats available for phone set: ";
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        ss << phone_sets[i][j] << ' ' ;
      KALDI_WARN  << ss.str();
    }
  }

  TreeClusterOptions topts;
  topts.kmeans_cfg.num_tries = 10;  // This is a slow-but-accurate setting,
  // we do it this way since there are typically few phones.

  std::vector<int32> assignments;  // assignment of phones to clusters. dim == summed_stats.size().
  std::vector<int32> clust_assignments;  // Parent of each cluster.  Dim == #clusters.
  int32 num_leaves;  // number of leaf-level clusters.
  TreeCluster(summed_stats_per_set,
              summed_stats_per_set.size(),  // max-#clust is all of the points.
              NULL,  // don't need the clusters out.
              &assignments,
              &clust_assignments,
              &num_leaves,
              topts);

  // process the information obtained by TreeCluster into the
  // form we want at output.
  ObtainSetsOfPhones(phone_sets,
                     assignments,
                     clust_assignments,
                     num_leaves,
                     questions_out);

  // The memory in summed_stats was newly allocated. [the other algorithms
  // used here do not allocate].
  DeletePointers(&summed_stats);
  DeletePointers(&summed_stats_per_set);

}


void KMeansClusterPhones(BuildTreeStatsType &stats,
                         const std::vector<std::vector<int32> > &phone_sets_in,
                         const std::vector<int32> &all_hmm_positions_in,
                         int32 P,
                         int32 num_classes,
                         std::vector<std::vector<int32> > *sets_out) {
  std::vector<std::vector<int32> > phone_sets(phone_sets_in);
  std::vector<int32> phones;
  for (size_t i = 0; i < phone_sets.size() ;i++) {
    std::sort(phone_sets[i].begin(), phone_sets[i].end());
    if (phone_sets[i].empty())
      KALDI_ERR << "Empty phone set in AutomaticallyObtainQuestions";
    if (!IsSortedAndUniq(phone_sets[i]))
      KALDI_ERR << "Phone set in AutomaticallyObtainQuestions contains duplicate phones";
    for (size_t j = 0; j < phone_sets[i].size(); j++)
      phones.push_back(phone_sets[i][j]);
  }
  std::sort(phones.begin(), phones.end());
  if (!IsSortedAndUniq(phones))
    KALDI_ERR << "Phones are present in more than one phone set.";
  if (phones.empty())
    KALDI_ERR << "No phones provided.";

  std::vector<int32> all_hmm_positions(all_hmm_positions_in);
  SortAndUniq(&all_hmm_positions);
  assert(!all_hmm_positions.empty());

  BuildTreeStatsType retained_stats;
  FilterStatsByKey(stats, kPdfClass, all_hmm_positions,
                   true,  // retain only the listed positions
                   &retained_stats);


  std::vector<BuildTreeStatsType> split_stats;  // split by phone.
  SplitStatsByKey(retained_stats, P, &split_stats);

  std::vector<Clusterable*> summed_stats;  // summed up by phone.
  SumStatsVec(split_stats, &summed_stats);

  int32 max_phone = phones.back();
  if (static_cast<int32>(summed_stats.size()) < max_phone+1) {
    // this can happen if the last phone had no data.. if we are using
    // stress-marked, position-marked phones, this can happen.  The later
    // code will assume that a summed_stats entry exists for all phones.
    summed_stats.resize(max_phone+1, NULL);
  }

  for (int32 i = 0; static_cast<size_t>(i) < summed_stats.size(); i++) {
    // just a check.
    if (summed_stats[i] != NULL &&
       !binary_search(phones.begin(), phones.end(), i)) {
      KALDI_WARN << "Phone "<< i << " is present in stats but is not in phone list [make sure you intended this].";
    }
  }

  EnsureClusterableVectorNotNull(&summed_stats);  // make sure no NULL pointers in summed_stats.
  // will replace them with pointers to empty stats.

  std::vector<Clusterable*> summed_stats_per_set(phone_sets.size(), NULL);  // summed up by set.
  for (size_t i = 0; i < phone_sets.size(); i++) {
    const std::vector<int32> &this_set = phone_sets[i];
    summed_stats_per_set[i] = summed_stats[this_set[0]]->Copy();
    for (size_t j = 1; j < this_set.size(); j++)
      summed_stats_per_set[i]->Add(*(summed_stats[this_set[j]]));
  }

  for (size_t i = 0; i < summed_stats_per_set.size(); i++) {  // A check.
    if (summed_stats_per_set[i]->Normalizer() == 0.0) {
      std::ostringstream ss;
      ss << "AutomaticallyObtainQuestions: no stats available for phone set: ";
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        ss << phone_sets[i][j] << ' ' ;
      KALDI_WARN  << ss.str();
    }
  }

  ClusterKMeansOptions opts;  // Just using the default options which are a reasonable
  // compromise between speed and accuracy.

  std::vector<int32> assignments;
  BaseFloat objf_impr = ClusterKMeans(summed_stats_per_set,
                                      num_classes,
                                      NULL,
                                      &assignments,
                                      opts);

  BaseFloat count = SumClusterableNormalizer(summed_stats_per_set);

  KALDI_LOG << "ClusterKMeans: objf change from clustering [versus single set] is "
            << (objf_impr/count) << " over " << count << " frames.";

  sets_out->resize(num_classes);
  KALDI_ASSERT(assignments.size() == phone_sets.size());
  for (size_t i = 0; i < assignments.size(); i++) {
    int32 class_idx = assignments[i];
    assert(static_cast<size_t>(class_idx) < sets_out->size());
    for (size_t j = 0; j < phone_sets[i].size(); j++)
      (*sets_out)[class_idx].push_back(phone_sets[i][j]);
  }
  for (size_t i = 0; i < sets_out->size(); i++) {
    std::sort( (*sets_out)[i].begin(), (*sets_out)[i].end() );  // just good
    // practice to have them sorted as who knows if whatever we need them for
    // will require sorting...
    KALDI_ASSERT(IsSortedAndUniq( (*sets_out)[i] ));
  }
  DeletePointers(&summed_stats);
  DeletePointers(&summed_stats_per_set);
}

void ReadRootsFile(std::istream &is,
                   std::vector<std::vector<int32> > *phone_sets,
                   std::vector<bool> *is_shared_root,
                   std::vector<bool> *is_split_root) {
  KALDI_ASSERT(phone_sets != NULL && is_shared_root != NULL &&
               is_split_root != NULL && phone_sets->empty()
               && is_shared_root->empty() && is_split_root->empty());

  std::string line;
  int line_number = 0;
  while ( ! getline(is, line).fail() ) {
    line_number++;
    std::istringstream ss(line);
    std::string shared;
    ss >> shared;
    if (ss.fail() && shared != "shared" && shared != "not-shared")
      KALDI_EXIT << "Bad line in roots file: line "<< line_number << ": " << line;
    is_shared_root->push_back(shared == "shared");

    std::string split;
    ss >> split;
    if (ss.fail() && shared != "split" && shared != "not-split")
      KALDI_EXIT << "Bad line in roots file: line "<< line_number << ": " << line;
    is_split_root->push_back(split == "split");

    phone_sets->push_back(std::vector<int32>());
    int32 i;
    while ( !(ss >> i).fail() ) {
      phone_sets->back().push_back(i);
    }
    std::sort(phone_sets->back().begin(), phone_sets->back().end());
    if (!IsSortedAndUniq(phone_sets->back()) || phone_sets->back().empty()
       || phone_sets->back().front() <= 0)
      KALDI_EXIT << "Bad line in roots file [empty, or contains non-positive "
                 << " or duplicate phone-ids]: line " << line_number << ": "
                 << line;
  }
  if (phone_sets->empty())
    KALDI_EXIT << "Empty roots file ";
}


/// Helper function for CreateUntiedTree and CreateUntiedTreeHelper
/// Return an EventMap that splits on pdf-id, for a particular
/// context.
EventMap *UntiedTreeHandleOneContext(const std::vector<int32> &context,
                                     int32 num_pdf_classes, 
                                     std::vector<std::vector<int32> > *states) {
  std::vector<EventMap*> leaves_per_pdf_class;
  KALDI_ASSERT(num_pdf_classes > 0);
  for (int32 i = 0; i < num_pdf_classes; i++) {
    std::vector<int32> this_state(context);
    this_state.push_back(i);
    int32 this_pdf_id = states->size();
    leaves_per_pdf_class.push_back(new ConstantEventMap(this_pdf_id));
    states->push_back(this_state);
  }
  // note: kPdfClass==-1 is the key corresponding to the pdf-class.
  return new TableEventMap(kPdfClass, leaves_per_pdf_class);
}

/// Recursive helper for CreateUntiedTree.  Splits on each of the
/// members of "keys_to_split_in".
EventMap *CreateUntiedTreeHelper(const BuildTreeStatsType &stats,
                                 const std::vector<int32> &keys_to_split_in,
                                 int32 num_pdf_classes,
                                 std::vector<std::vector<int32> > *states) {
  std::vector<int32> keys_to_split(keys_to_split_in);
  if (stats.empty()) return NULL;
  else if (stats.size() == 1) { // no point splitting on the specified keys,
    // just split on the pdf-class.
    std::vector<EventMap*> split_on_pdf_class;
    std::vector<int32> context;
    const EventType &evec = stats[0].first;
    for (int32 j = 0; j < static_cast<int32>(evec.size()); j++) {
      KALDI_ASSERT(evec[j].first == j);
      context.push_back(evec[j].second);
    }
    return UntiedTreeHandleOneContext(context, num_pdf_classes, states);
  } else {
    KALDI_ASSERT(!keys_to_split.empty()); // or how could there be >1 stats?
    int32 key_to_split = keys_to_split.back();
    keys_to_split.pop_back();
    std::vector<BuildTreeStatsType> split_stats;
    SplitStatsByKey(stats, key_to_split, &split_stats);
    std::vector<EventMap*> ans_maps(split_stats.size(), NULL);
    for (size_t i = 0; i < ans_maps.size(); i++) {
      ans_maps[i] = CreateUntiedTreeHelper(split_stats[i],
                                          keys_to_split,
                                          num_pdf_classes,
                                          states);
    }
    return new TableEventMap(key_to_split, ans_maps);
  }
}


EventMap *CreateUntiedTree(int N, int P,
                           const std::vector<std::vector<int32> > &contexts,
                           const std::vector<int32> &phone2num_pdf_classes,
                           const std::vector<int32> &ci_phones_in,
                           std::vector<std::vector<int32> > *states) {
  KALDI_ASSERT(!contexts.empty());
  states->clear();
  for (size_t i = 0; i < contexts.size(); i++)
    KALDI_ASSERT(contexts[i].size() == static_cast<size_t>(N));

  // It's convenient to represent the context as "fake stats"
  // consisting of pairs of ( (sets of (key,value) pairs), Clusterable* ),
  // where the Clusterable* pointers are NULL.  This means we
  // can use functions like SplitsStatsByKey.
  BuildTreeStatsType fake_stats;
  for (size_t i = 0; i < contexts.size(); i++) {
    EventType evec;
    for (int32 j = 0; j < N; j++) {
      int32 key = j, value = contexts[i][j];
      evec.push_back(std::make_pair<EventKeyType, EventValueType>(key, value));
    }
    Clusterable *cptr = NULL;
    fake_stats.push_back(std::make_pair(evec, cptr));
  }
  
  std::vector<BuildTreeStatsType> stats_per_phone;
  SplitStatsByKey(fake_stats, P, &stats_per_phone);
  std::vector<EventMap*> map_per_phone(stats_per_phone.size(), NULL);
  KALDI_ASSERT(stats_per_phone[0].empty());
  for (int32 p = 1; p < static_cast<int32>(stats_per_phone.size()); p++) {
    if (!stats_per_phone[p].empty()) {
      KALDI_ASSERT(static_cast<size_t>(p) < phone2num_pdf_classes.size());
      int32 num_pdf_classes = phone2num_pdf_classes[p];
      KALDI_ASSERT (num_pdf_classes > 0);
      
      if (std::find(ci_phones_in.begin(), ci_phones_in.end(), p) !=
          ci_phones_in.end()) { // it is a context-indep. phone.
        if (stats_per_phone[p].size() != 1u)
          KALDI_ERR << "Error: you specified more than one phonetic context for "
                    << "a context-independent phone.";
        
        const EventType &evec(stats_per_phone[p][0].first);
        std::vector<int32> context;
        for (int32 j = 0; j < static_cast<int32>(evec.size()); j++) {
          KALDI_ASSERT(evec[j].first == j);
          context.push_back(evec[j].second);
        }
        map_per_phone[p] = UntiedTreeHandleOneContext(context,
                                                      num_pdf_classes,
                                                      states);
      } else {
        // not context-dep.
        // split on everything but the central phone...
        std::vector<int32> keys;
        for (int32 i = 0; i < N; i++)
          if (i != P) keys.push_back(i);
        map_per_phone[p] = CreateUntiedTreeHelper(stats_per_phone[p], keys,
                                                  num_pdf_classes, states);
      }
    }
  }
  return new TableEventMap(P, map_per_phone);
}



} // end namespace kaldi
