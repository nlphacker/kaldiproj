// cudamatrix/cu-device.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CU_DEVICE_H_
#define KALDI_CUDAMATRIX_CU_DEVICE_H_

#if HAVE_CUDA==1

#include <map>
#include <string>
#include <iostream>

namespace kaldi {

/**
 * Singleton object which represents CUDA device
 * responsible for CUBLAS initilalisation, collects profiling info
 */
class CuDevice {
 // Singleton interface...
 private:
  CuDevice();
  CuDevice(CuDevice&);
  CuDevice &operator=(CuDevice&);

 public:
  ~CuDevice();
  static CuDevice& Instantiate() { 
    return msDevice; 
  }

 private:
  static CuDevice msDevice;


 /**********************************/
 // Instance interface
 public:
 
  /// Check if the CUDA device is selected for use
  bool Enabled() { 
    return (active_gpu_id_ > -1); 
  }

  /// Manually select GPU by id (more comments in cu-device.cc)
  void SelectGpuId(int32 gpu_id);
  /// Get the active GPU id
  int32 ActiveGpuId() {
    return active_gpu_id_;
  }

  void Verbose(bool verbose) { 
    verbose_ = verbose; 
  }

  /// Sum the IO time
  void AccuProfile(const std::string &key, double time);
  void PrintProfile(); 
  
  void ResetProfile() { 
    profile_map_.clear(); 
  }
  
  /// Get the actual GPU memory use stats
  std::string GetFreeMemory(int64* free = NULL, int64* total = NULL);
  /// Get the name of the GPU
  void DeviceGetName(char* name, int32 len, int32 dev); 
  
 private:
  /// Check if the GPU run in compute exclusive mode
  bool IsComputeExclusive();
  /// Automatically select GPU
  void SelectGpuIdAuto();

 private:
  std::map<std::string, double> profile_map_;
  
  /// active_gpu_id_ values:
  /// -3 default (default, the SelectGpuId was not called, we did not want to use GPU)
  /// -2 SelectGpuId was called, but no GPU was present
  /// -1 SelectGpuId was called, but the GPU was manually disabled
  /// 0..N Normal GPU IDs
  int32 active_gpu_id_; 
  ///

  bool verbose_;

}; // class CuDevice


}// namespace

#endif // HAVE_CUDA


#endif
