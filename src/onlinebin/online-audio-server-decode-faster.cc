// onlinebin/online-audio-server-decode-faster.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Copyright 2013 Polish-Japanese Institute of Information Technology (author: Danijel Korzinek)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "online/online-tcp-source.h"
#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"
#include "matrix/kaldi-vector.h"
#include "lat/word-align-lattice.h"
#include "lat/lattice-functions.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>

using namespace kaldi;
using namespace fst;

/*
 * This class is for a very simple TCP server implementation
 * in UNIX sockets.
 */
class TCPServer {
 private:
  struct sockaddr_in h_addr;
  int32 server_desc;

 public:
  TCPServer();
  ~TCPServer();

  bool Listen(int32 port);  //start listening on a given port
  int32 Accept();  //accept a client and return its descriptor
};

//write a line of text to socket
bool WriteLine(int32 socket, std::string line);

//constant allowing to convert frame count to time
const float kFramesPerSecond = 100.0f;

int32 main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef OnlineFeInput<OnlineTCPVectorSource, Mfcc> FeInput;
    TCPServer tcp_server;

    // up to delta-delta derivative features are calculated (unless LDA is used)
    const int32 kDeltaOrder = 2;

    const char *usage =
        "Starts a TCP server that receives RAW audio and outputs aligned words.\n"
            "A sample client can be found in: onlinebin/online-audio-client\n\n"
            "Usage: ./online-audio-server-decode-faster [options] model-in"
            "fst-in word-symbol-table silence-phones tcp-port word-boundary-file lda-matrix-in\n\n"
            "word-boundary file is a file that maps phoneme ids to one of (nonword|begin|end|internal|single)\n\n"
            "example: online-audio-server-decode-faster --verbose=1 --rt-min=0.5 --rt-max=3.0 --max-active=6000\n"
            "--beam=72.0 --acoustic-scale=0.0769 final.mdl graph/HCLG.fst graph/words.txt '1:2:3:4:5' 5010\n"
            "graph/word_boundary_phones.txt final.mat\n\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;
    int32 cmn_window = 600, min_cmn_window = 100;  // adds 1 second latency, only at utterance start.
    int32 right_context = 4, left_context = 4;

    OnlineFasterDecoderOpts decoder_opts;
    decoder_opts.Register(&po, true);
    OnlineFeatureMatrixOptions feature_reading_opts;
    feature_reading_opts.Register(&po);

    po.Register("left-context", &left_context,
                "Number of frames of left context");
    po.Register("right-context", &right_context,
                "Number of frames of right context");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register(
        "cmn-window", &cmn_window,
        "Number of feat. vectors used in the running average CMN calculation");
    po.Register("min-cmn-window", &min_cmn_window,
                "Minumum CMN window used at start of decoding (adds "
                "latency only at start)");

    WordBoundaryInfoNewOpts opts;
    opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 7) {
      po.PrintUsage();
      return 1;
    }

    if (left_context % kDeltaOrder != 0 || left_context != right_context)
      KALDI_ERR<< "Invalid left/right context parameters!";

    std::string model_rspecifier = po.GetArg(1), fst_rspecifier = po.GetArg(2),
        word_syms_filename = po.GetArg(3), silence_phones_str = po.GetArg(4),
        word_boundary_filename = po.GetOptArg(6), lda_mat_rspecifier = po
            .GetOptArg(7);

    int32 port = strtol(po.GetArg(5).c_str(), 0, 10);

    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR<< "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
      KALDI_ERR<< "No silence phones given!";

    if (!tcp_server.Listen(port))
      return 0;

    std::cout << "Reading LDA matrix: " << lda_mat_rspecifier << "..."
              << std::endl;
    Matrix<BaseFloat> lda_transform;
    if (lda_mat_rspecifier != "") {
      bool binary_in;
      Input ki(lda_mat_rspecifier, &binary_in);
      lda_transform.Read(ki.Stream(), binary_in);
    }

    std::cout << "Reading acoustic model: " << model_rspecifier << "..."
              << std::endl;
    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rspecifier, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    std::cout << "Reading word list: " << word_syms_filename << "..."
              << std::endl;
    fst::SymbolTable *word_syms = NULL;
    if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
      KALDI_ERR<< "Could not read symbol table from file "
      << word_syms_filename;

    std::cout << "Reading word boundary file: " << word_boundary_filename
              << "..." << std::endl;
    WordBoundaryInfo info(opts, word_boundary_filename);

    std::cout << "Reading FST: " << fst_rspecifier << "..." << std::endl;
    fst::Fst < fst::StdArc > *decode_fst = ReadDecodeGraph(fst_rspecifier);

    // We are not properly registering/exposing MFCC and frame extraction options,
    // because there are parts of the online decoding code, where some of these
    // options are hardwired(ToDo: we should fix this at some point)
    MfccOptions mfcc_opts;
    mfcc_opts.use_energy = false;
    int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
    int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

    int32 window_size = right_context + left_context + 1;
    decoder_opts.batch_size = std::max(decoder_opts.batch_size, window_size);

    VectorFst<LatticeArc> out_fst;
    CompactLattice out_lat, aligned_lat;
    OnlineTCPVectorSource* au_src = NULL;
    int32 client_socket = -1;

    while (true) {
      if (au_src == NULL || !au_src->IsConnected()) {
        if (au_src) {
          std::cout << "Client disconnected!" << std::endl;
          delete au_src;
        }
        client_socket = tcp_server.Accept();
        au_src = new OnlineTCPVectorSource(client_socket);
      }

      //re-initalizing decoder for each utterance
      OnlineFasterDecoder decoder(*decode_fst, decoder_opts, silence_phones,
                                  trans_model);

      Mfcc mfcc(mfcc_opts);
      FeInput fe_input(au_src, &mfcc, frame_length * (16000 / 1000),
                       frame_shift * (16000 / 1000));  //we always assume 16 kHz Fs on input
      OnlineCmnInput cmn_input(&fe_input, cmn_window, min_cmn_window);
      OnlineFeatInputItf *feat_transform = 0;
      if (lda_mat_rspecifier != "") {
        feat_transform = new OnlineLdaInput(&cmn_input, lda_transform,
                                            left_context, right_context);
      } else {
        DeltaFeaturesOptions opts;
        opts.order = kDeltaOrder;
        // Note from Dan: keeping the next statement for back-compatibility,
        // but I don't think this is really the right way to set the window-size
        // in the delta computation: it should be a separate config.
        opts.window = left_context / 2;
        feat_transform = new OnlineDeltaInput(opts, &cmn_input);
      }

      // feature_reading_opts contains timeout, batch size.
      OnlineFeatureMatrix feature_matrix(feature_reading_opts, feat_transform);

      OnlineDecodableDiagGmmScaled decodable(am_gmm, trans_model,
                                             acoustic_scale, &feature_matrix);

      clock_t start = clock();
      int32 decoder_offset = 0;

      while (1) {
        if (!au_src->IsConnected())
          break;

        OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);

        if (!au_src->IsConnected()) {
          break;
        }

        if (dstate & (decoder.kEndFeats | decoder.kEndUtt)) {
          std::vector<int32> word_ids, times, lengths;

          decoder.FinishTraceBack(&out_fst);
          decoder.GetBestPath(&out_fst);

          ConvertLattice(out_fst, &out_lat);

          WordAlignLattice(out_lat, trans_model, info, 0, &aligned_lat);

          CompactLatticeToWordAlignment(aligned_lat, &word_ids, &times,
                                        &lengths);

          //count number of non-sil words
          int32 words_num = 0;
          for (size_t i = 0; i < word_ids.size(); i++)
            if (word_ids[i] != 0)
              words_num++;

          if (words_num > 0) {

            float dur = (clock() - start) / (float) CLOCKS_PER_SEC;
            float input_dur = au_src->SamplesProcessed() / 16000.0;

            start = clock();
            au_src->ResetSamples();

            std::stringstream sstr;
            sstr << "RESULT:NUM=" << words_num << ",FORMAT=WSE,RECO-DUR=" << dur
                                               << ",INPUT-DUR=" << input_dur;

            WriteLine(client_socket, sstr.str());

            for (size_t i = 0; i < word_ids.size(); i++) {
              if (word_ids[i] == 0)
                continue;  //skip silences...

              std::string word = word_syms->Find(word_ids[i]);
              if (word.empty())
                word = "???";

              float start = (times[i] + decoder_offset) / kFramesPerSecond;
              float len = lengths[i] / kFramesPerSecond;

              std::stringstream wstr;
              wstr << word << "," << start << "," << (start + len);

              WriteLine(client_socket, wstr.str());
            }
          }

          if (dstate == decoder.kEndFeats) {
            WriteLine(client_socket, "RESULT:DONE");
            break;
          }

          decoder_offset = decoder.frame();
        }
      }
      if (feat_transform)
        delete feat_transform;
    }

    std::cout << "Deinitizalizing..." << std::endl;

    if (word_syms)
      delete word_syms;
    if (decode_fst)
      delete decode_fst;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}  // main()

// IMPLEMENTATION OF THE CLASSES/METHODS ABOVE MAIN
TCPServer::TCPServer() {
  server_desc = -1;
}

bool TCPServer::Listen(int32 port) {
  h_addr.sin_addr.s_addr = INADDR_ANY;
  h_addr.sin_port = htons(port);
  h_addr.sin_family = AF_INET;

  server_desc = socket(AF_INET, SOCK_STREAM, 0);

  if (server_desc == -1) {
    KALDI_ERR<< "Cannot create TCP socket!";
    return false;
  }

  if (bind(server_desc, (struct sockaddr*) &h_addr, sizeof(h_addr)) == -1) {
    KALDI_ERR<< "Cannot bind to port: "<<port<<" (is it taken?)";
    return false;
  }

  if (listen(server_desc, 1) == -1) {
    KALDI_ERR<< "Cannot listen on port!";
    return false;
  }

  std::cout << "TCPServer: Listening on port: " << port << std::endl;

  return true;

}

TCPServer::~TCPServer() {
  if (server_desc != -1)
    close(server_desc);
}

int32 TCPServer::Accept() {
  std::cout << "Waiting for client..." << std::endl;

  socklen_t len;

  int32 client_desc = accept(server_desc, (struct sockaddr*) &h_addr, &len);

  struct sockaddr_storage addr;
  char ipstr[20];

  len = sizeof addr;
  getpeername(client_desc, (struct sockaddr*) &addr, &len);

  struct sockaddr_in *s = (struct sockaddr_in *) &addr;
  inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);

  std::cout << "TCPServer: Accepted connection from: " << ipstr << std::endl;

  return client_desc;
}

bool WriteLine(int32 socket, std::string line) {
  line = line + "\n";

  const char* p = line.c_str();
  int32 to_write = line.size();
  int32 wrote = 0;
  while (to_write > 0) {
    int32 ret = write(socket, p + wrote, to_write);
    if (ret <= 0)
      return false;

    to_write -= ret;
    wrote += ret;
  }

  return true;
}
