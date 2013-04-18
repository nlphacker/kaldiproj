#!/bin/bash 
set -e
set -o pipefail
set -u

type=dev2h

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;


type=dev10h
. utils/parse_options.sh     


if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) [--type (dev10h|dev2h|eval|shadow)]"
fi

if [[ "$type" != "dev10h" && "$type" != "dev2h" && "$type" != "eval" && "$type" != "shadow" ]]; then
  echo "Error: invalid variable type=$type, valid values are dev10h|dev2h|eval|shadow"
fi

# Yenda: please check if this logic is right.  Not sure what to do about shadow, and about
# kws scoring versus CTM scoring.
# Also, I might not be passing this variable in everywhere I need to.
if [[ "$type" == "dev2h" || "$type" == "dev10h" ]]; then
  skip_scoring=false
else
  skip_scoring=true
fi


[ ! -f $eval_data_cmudb ] && echo "The CMU db does not exist... " && exit 1
#[ ! -f $eval_data_list ] && echo "The test data list does not exist... " && exit 1
#[ ! -d $eval_data_dir ] && echo "The test data directory does not exist... " && exit 1

[ -z $cer ] && cer=0

#####################################################################
#
# test.uem and shadow.uem directory preparation
#
#####################################################################
if [[ $type == shadow || $type == eval ]]; then
  if [ ! -f data/eval.uem/.done ]; then
    local/cmu_uem2kaldi_dir.sh --filelist $eval_data_list $eval_data_cmudb_shadow  $eval_data_dir data/eval.uem 
    steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/eval.uem exp/make_plp/eval.uem plp

    steps/compute_cmvn_stats.sh data/eval.uem exp/make_plp/eval.uem plp
    utils/fix_data_dir.sh data/eval.uem
    touch data/eval.uem/.done
  fi

  # we expect that the dev2h directory will already exist.
  if [ ! -f data/shadow.uem/.done ]; then
    if [ ! -f data/dev2h/.done ]; then
      echo "Error: data/dev2h/.done does not exist."
    fi

    local/kws_setup.sh --case-insensitive $case_insensitive $eval_data_ecf $eval_data_kwlist data/lang data/eval.uem/

    local/create_shadow_dataset.sh data/shadow.uem data/dev2h data/eval.uem
    local/kws_data_prep.sh --case-insensitive $case_insensitive data/lang data/shadow.uem data/shadow.uem/kws || exit 1
    utils/fix_data_dir.sh data/shadow.uem
    touch data/shadow.uem/.done 
  fi
fi

#####################################################################
#
# dev10h.uem directory preparation
#
#####################################################################
if [ $type == dev10h ]; then
  dev10h_data_cmudb=$eval_data_cmudb
  dev10h_ecf_file=$ecf_file
  dev10h_kwlist_file=$kwlist_file
  dev10h_rttm_file=$rttm_file
  decode_nj=$dev10h_nj
  if [ ! -f data/dev10h.uem/.done ]; then
    local/cmu_uem2kaldi_dir.sh --filelist $dev10h_data_list $dev10h_data_cmudb  $dev10h_data_dir data/dev10h.uem
    cp data/dev10h/stm data/dev10h.uem
    cp data/dev10h/glm data/dev10h.uem
    if [ $cer == 1 ] ; then
      [ ! -f data/dev10h/char.stm ] && echo "CER=1 and file dev10h/char.stm not present" && exit 1
      cp data/dev10h/char.stm data/dev10h.uem
    fi
    steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/dev10h.uem exp/make_plp/dev10h.uem plp
    steps/compute_cmvn_stats.sh data/dev10h.uem exp/make_plp/dev10h.uem plp
    utils/fix_data_dir.sh data/dev10h.uem

    local/kws_setup.sh --case-insensitive $case_insensitive \
      --rttm-file $dev10h_rttm_file \
      $dev10h_ecf_file $dev10h_kwlist_file data/lang data/dev10h.uem
    touch data/dev10h.uem/.done
  fi
fi

if [ $type == dev2h ]; then
  dev2h_data_cmudb=$eval_data_cmudb
  dev2h_ecf_file=$ecf_file
  dev2h_kwlist_file=$kwlist_file
  dev2h_rttm_file=$rttm_file
  if [ ! -f data/dev2h.uem/.done ]; then
    local/cmu_uem2kaldi_dir.sh --filelist $dev2h_data_list $dev2h_data_cmudb  $dev2h_data_dir data/dev2h.uem
    cp data/dev2h/stm data/dev2h.uem
    cp data/dev2h/glm data/dev2h.uem
    if [ $cer == 1 ] ; then
      [ ! -f data/dev2h/char.stm ] && echo "CER=1 and file eval.pem/char.stm not present" && exit 1
      cp data/dev2h/char.stm data/dev2h.uem
    fi
    steps/make_plp.sh --cmd "$train_cmd" --nj $decode_nj data/dev2h.uem exp/make_plp/dev2h.uem plp
    steps/compute_cmvn_stats.sh data/dev2h.uem exp/make_plp/dev2h.uem plp
    utils/fix_data_dir.sh data/dev2h.uem

    local/kws_setup.sh --case-insensitive $case_insensitive \
      --subset-ecf $dev2h_data_list --rttm-file $dev2h_rttm_file \
      $dev2h_ecf_file $dev2h_kwlist_file data/lang data/dev2h.uem
    touch data/dev2h.uem/.done
  fi
  decode_nj=$decode_nj
fi

####################################################################
##
## FMLLR decoding 
##
####################################################################



if [ ! -f exp/tri5/decode_${type}.uem/.done ]; then
  [ ! -d exp/tri5/graph ] && utils/mkgraph.sh data/lang exp/tri5 exp/tri5/graph |tee exp/tri5/mkgraph.log

  steps/decode_fmllr_extra.sh --skip-scoring "$skip_scoring" \
    --nj $decode_nj --cmd "$decode_cmd"  "${decode_extra_opts[@]}"\
    exp/tri5/graph data/${type}.uem exp/tri5/decode_${type}.uem  | tee  exp/tri5/decode_${type}.uem.log
  touch exp/tri5/decode_${type}.uem/.done
fi


if [ ! -f exp/tri5/decode_${type}.uem/.kws.done ]; then
  if [ $type == shadow ]; then
    local/lattice_to_ctm.sh --cmd "$decode_cmd" data/${type}.uem data/lang exp/tri5/decode_${type}.uem.si
    local/lattice_to_ctm.sh --cmd "$decode_cmd" data/${type}.uem data/lang exp/tri5/decode_${type}.uem
    local/split_ctms.sh exp/tri5/decode_shadow.uem.si data/dev2h data/eval.uem
    local/split_ctms.sh exp/tri5/decode_shadow.uem data/dev2h data/eval.uem
    
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/${type}.uem data/lang \
      exp/tri/decode_${type}.uem.si data/dev2h data/eval.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/${type}.uem data/lang \
      exp/tri5/decode_${type}.uem data/dev2h data/eval.uem
  else
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/tri5/decode_${type}.uem.si
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/tri5/decode_${type}.uem
  fi
  touch exp/tri5/decode_${type}.uem/.kws.done
fi

####################################################################
## SGMM2 decoding 
####################################################################
if [ ! -f exp/sgmm5/decode_fmllr_${type}.uem/.done ]; then
  [ ! -d exp/sgmm5/graph ] && utils/mkgraph.sh data/lang exp/sgmm5 exp/sgmm5/graph |tee exp/sgmm5/mkgraph.log
  steps/decode_sgmm2.sh --skip-scoring "$skip_scoring" --use-fmllr true --nj $decode_nj \
    --cmd "$decode_cmd" --transform-dir exp/tri5/decode_${type}.uem "${decode_extra_opts[@]}"\
    exp/sgmm5/graph data/${type}.uem exp/sgmm5/decode_fmllr_${type}.uem | tee exp/sgmm5/decode_fmllr_${type}.uem.log
  touch exp/sgmm5/decode_fmllr_${type}.uem/.done
fi  

if [ ! -f exp/sgmm5/decode_fmllr_${type}.uem/.kws.done ]; then
  local/lattice_to_ctm.sh --cmd "$decode_cmd" data/$type.uem data/lang exp/sgmm5/decode_fmllr_$type.uem 
  if [ $type == shadow ]; then
    local/split_ctms.sh exp/sgmm5/decode_fmllr_shadow.uem data/dev2h data/eval.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" data/shadow.uem data/lang \
      exp/sgmm5/decode_fmllr_shadow.uem data/dev2h data/eval.uem
  else
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/sgmm5/decode_fmllr_${type}.uem
  fi
  touch exp/sgmm5/decode_fmllr_${type}.uem/.kws.done
fi

####################################################################
##
## SGMM_MMI rescoring and KWS
##
####################################################################

for iter in 1 2 3 4; do
  # Decode SGMM+MMI (via rescoring).
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.done ]; then

    steps/decode_sgmm2_rescore.sh  --skip-scoring "$skip_scoring" \
      --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_${type}.uem \
      data/lang data/${type}.uem exp/sgmm5/decode_fmllr_${type}.uem exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter
   
    touch exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.done
  fi
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.kws.done ]; then
    #local/lattice_to_ctm.sh --cmd "$decode_cmd" --word-ins-penalty 0.5 \
    #  data/${type}.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter

    if [ $type == shadow ]; then
      local/split_ctms.sh data/shadow.uem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
        data/dev2h data/eval.uem
      local/shadow_set_kws_search.sh --cmd "$decode_cmd" --max-states 150000 \
        data/shadow.uem data/lang exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.uem_it$iter \
        data/dev2h data/eval.uem
    else
      local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
        data/${type}.uem/ exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter
    fi
    touch exp/sgmm5_mmi_b0.1/decode_fmllr_${type}.uem_it$iter/.kws.done
  fi
done


if [[ ! -f exp/tri6_nnet/decode_${type}.uem/.done && -f exp/tri6_nnet/final.mdl ]]; then

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
    --skip-scoring "$skip_scoring" "${decode_extra_opts[@]}" \
    --transform-dir exp/tri5/decode_${type}.uem/ \
    exp/tri5/graph data/${type}.uem exp/tri6_nnet/decode_${type}.uem
    
  touch exp/tri6_nnet/decode_${type}.uem/.done
fi

if [[ ! -f exp/tri6_nnet/decode_${type}.uem/.kws.done && -f exp/tri6_nnet/final.mdl ]]; then

  #local/lattice_to_ctm.sh --cmd "$decode_cmd" --word-ins-penalty 0.5 \
  #  data/${type}.uem data/lang exp/tri6_nnet/decode_${type}.uem 

  if [ "$type" != eval ]; then
    local/score_stm.sh --cmd "$decode_cmd" \
      data/${type}.uem data/lang exp/tri6_nnet/decode_${type}.uem
  fi

  if [ $type == shadow ]; then
    local/split_ctms.sh data/shadow.uem exp/tri6_nnet/decode_${type}.uem \
      data/dev2h data/eval.uem
    local/shadow_set_kws_search.sh --cmd "$decode_cmd" --max-states 150000 \
      data/shadow.uem data/lang exp/tri6_nnet/decode_${type}.uem \
      data/dev2h data/eval.uem
  else
    local/kws_search.sh --skip-scoring "$skip_scoring" --cmd "$decode_cmd" data/lang \
      data/${type}.uem/ exp/tri6_nnet/decode_${type}.uem
  fi
  touch exp/tri6_nnet/decode_${type}.uem/.kws.done
fi


echo "Everything looks fine"
