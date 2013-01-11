#!/bin/bash

# WARNING to all those who enter here.
# This setup works OK on RM, but has not worked well on other
# setups-- it's been as much as 10% absolute worse than Karel's setup.
# We think that the issue is, the automatic setting of learning-rates is
# setting them lower than they should be.  We're working on it.
# Just be aware of the issue.

. cmd.sh

steps/train_nnet_cpu.sh data/train data/lang exp/tri3b_ali exp/tri4b_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4b_nnet/decode

for iter in 1 2 3 4; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd -pe smp 5" --nj 20 \
    --iter $iter \
    --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4b_nnet/decode_it$iter &
done


steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --acwt 0.2 --beam 30.0 --lat-beam 15.0 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4b_nnet/decode2



steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
   data/train data/lang exp/tri3b_ali exp/tri4c_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4c_nnet/decode


for iter in 1 2 3 4; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd -pe smp 5" --nj 20 \
    --iter $iter \
    --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4c_nnet/decode_it$iter &
done

# should be like 4c, just doing it with a newer script that
# is more configurable.
steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
   data/train data/lang exp/tri3b_ali exp/tri4d_nnet

# should be same as 4c.
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4d_nnet/decode

( #1m parameters.
  steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4e_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4e_nnet/decode
)

( 
  steps/train_nnet_cpu_block.sh --cmd "queue.pl -pe smp 15" \
   --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4f_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
    --transform-dir exp/tri3b/decode \
    exp/tri3b/graph data/test exp/tri4f_nnet/decode
)


( 
  steps/train_nnet_cpu_block.sh --cmd "queue.pl -pe smp 15" \
   --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4g_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
    --transform-dir exp/tri3b/decode \
    exp/tri3b/graph data/test exp/tri4g_nnet/decode
)

( # as 4e (1m parameters), but realigning on the 2nd iter.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "2" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4h_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4h_nnet/decode
)

( # as 4h but realigning on the 4th iter, not the 2nd.
  # Realigning does not really seem to help.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4i_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4i_nnet/decode
)


( # 4j is as 4i, but running again after I changed the way the
  # validation set is selected (more utts, subset of frames.)
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4j_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4j_nnet/decode
)

( # 4k is as 4j, but with --learning-rate-ratio=1.0 so it never decreases 
  # learning rate.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --learning-rate-ratio 1.0 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4k_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4k_nnet/decode
)

( # 4l is as 4j, but with --measure-gradient-at 0.6 for faster learning.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.6 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4l_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4l_nnet/decode
)


( # 4m is as 4l, but --measure-gradient-at now 0.8 not 0.6.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4m_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4m_nnet/decode
)

( # 4n is as 4l, but --measure-gradient-at now 0.55 not 0.6.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.55 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4n_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4n_nnet/decode
)

( # 4o is as 4m, but changing initial l2 penalty to 0.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4o_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4o_nnet/decode
)
( # 4o2 is as 4o, but actually doing what we said we were doing (fixed bug)
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4o2_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4o2_nnet/decode
)

( # 4p is as 4o (no l2 penalty, which is actually how all the previous expts were
  # but now making it explicit as we may have fixed the bug which caused that)
  # *and* smaller minibatch size; changing minibatches-per-phase to compensate.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4p_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4p_nnet/decode
)


( # 4q is as 4p, but the binaries and script are now changed so bias-stddev is 1.0.
 # [not replicable now, use --bias-stddev 1.0]
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4q_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4q_nnet/decode
)

( # 4r is as 4q, but the binaries and script are now changed so bias-stddev is 2.0 by default.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4r_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4r_nnet/decode
)

( # 4s is as 4r but bias-stddev is 4.0 not 2.0.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0 --bias-stddev 4.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4s_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4s_nnet/decode
)

( # 4t is as 4r but rerunning-- code should not have changed but we
 # want to check. [yes, it was the same.]
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4t_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4t_nnet/decode
)

( # 4v is as 4u but with code changed to do it more "properly".. previously was not working.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0 --precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4v_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4v_nnet/decode
)


( # 4w is as 4v but rerunning after I changed the code to remove l2 regularization;
  # just checking it still runs.
  steps/train_nnet_cpu.sh --nnet-config-opts "--precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4w_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4w_nnet/decode
)

( # running more iters on 4w.
  steps/train_nnet_cpu.sh --num-iters 20 --stage 5 \
  --nnet-config-opts "--precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4w_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4w_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4w_nnet/decode_it15
)

( # Trying preconditioned update.. this is different from the previous preconditioning which
  # was just bias removal.  Baseline is probably 4w.
  steps/train_nnet_cpu.sh --alpha 0.1 \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4x_nnet

)

# version of 4x that has alpha=1.0
( # preconditioned update with higher learning rate.
  steps/train_nnet_cpu.sh --alpha 1.0 \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4x2_nnet
)
# 4x3 is as 4x2 but with larger minibatch size.
( 
  steps/train_nnet_cpu.sh --alpha 1.0 \
    --minibatch-size 500 --minibatches-per-phase-it1 100 --minibatches-per-phase 400 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4x3_nnet
)
# 4x4 is as 4x3 but with even larger minibatch size.
( 
  steps/train_nnet_cpu.sh --alpha 1.0 \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4x4_nnet
)


( # variant of 4y that has higher alpha value, 0.25 vs 0.1
  steps/train_nnet_cpu.sh --alpha 0.25 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y2_nnet
)

( # variant of 4y that has even higher alpha value, 1.0 vs 0.1
  steps/train_nnet_cpu.sh --alpha 1.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y3_nnet
)

( # 4y4 is as 4y3 but larger minibatch size (1000)
  steps/train_nnet_cpu.sh --alpha 1.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y4_nnet
)

( # 4y5 is as 4y4 but larger alpha (4.0)
  steps/train_nnet_cpu.sh --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y5_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y5_nnet/decode_it5

  # more iters
  steps/train_nnet_cpu.sh --stage 5 --num-iters 10 --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y5_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y5_nnet/decode_it10

  # shrinkage.
  nnet-shrink exp/tri4y5_nnet/10.mdl ark:exp/tri4y5_nnet/valid.egs exp/tri4y5_nnet/10shrunk.mdl

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --iter 10shrunk \
     --transform-dir exp/tri3b/decode \
    exp/tri3b/graph data/test exp/tri4y5_nnet/decode_it10shrunk

)

( # 4y6 is as 4y5 but adding shrinkage.
  steps/train_nnet_cpu.sh --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y6_nnet
)

( # 4y7 is as 4y6 but fixing something-- continuing to update of
  # validation objf change is negative.  Also doing more iterations (15).
  steps/train_nnet_cpu.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y7_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y7_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y7_nnet/decode_it15
)

( # 4y8 is as 4y7 but using alpha=1.0
  steps/train_nnet_cpu.sh --num-iters 15 --shrink true --alpha 1.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y8_nnet
)

( # 4y9 is as 4y7 but double the learning rate.
  steps/train_nnet_cpu.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.005" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y9_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y9_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y9_nnet/decode_it15
)


( # 4y10 is as 4y7, but more recent code in which there's LDA after
  # splicing +- 4 frames..
  steps/train_nnet_cpu.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
   --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y10_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y10_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y10_nnet/decode_it15

)


( # 4y11 is as 4y10 (also splicing +- 4 frames), but with the "parallel" mode.
  # Note: the learning rate will not be updated in this code (we'll add this
  # option later).

  steps/train_nnet_cpu_parallel.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y11_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y11_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y11_nnet/decode_it15

)


( # 4y12 is as 4y11 (parallel mode), but added the automatic updating of
  # learning rates.  Note: this seemed to hurt.
  steps/train_nnet_cpu_parallel.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y12_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y12_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y12_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y12_nnet/decode_it15

)


(  # Broken-- 4y13 was identical to 4y11, gave slightly diff. results for some reason.
   # old comment:
   # 4y13 is as 4y11 (parallel mode), but removed the automatic updating of
  # learning rates and instead adding (where possible) the previous iteration's model
  # into the space we optimize over.
  steps/train_nnet_cpu_parallel2.sh --num-iters 10 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y13_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y13_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y13_nnet/decode_it10
)

(
   # 4y14 is as 4y11 (parallel mode), but removed the automatic updating of
   # learning rates and instead adding (where possible) the previous iteration's model
   # into the space we optimize over.
  steps/train_nnet_cpu_parallel2.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y14_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y14_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y14_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y14_nnet/decode_it15
)

(
   # 4y15 is as 4y14 (no automatic learning-rates updating) with a script
   # change so randomization can go in parallel with training.
  steps/train_nnet_cpu_parallel3.sh --num-iters 15 --shrink true --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y15_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y15_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y15_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y15_nnet/decode_it15
)

(
   # 4y16 is as 4y15 (could actually compare with some even
   # earlier scripts, but with the "parallel5" script which simplifies
   # the space we optimize over and does the learning-rate
   # updating in a different way.

  steps/train_nnet_cpu_parallel5.sh --num-iters 15  --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y16_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y16_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y16_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y16_nnet/decode_it15
)


(
   # 4y17 is as 4y16 but setting overshoot to 1.0 (no overshoot)

  steps/train_nnet_cpu_parallel5.sh --overshoot 1.0 --num-iters 15  --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y17_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 5 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y17_nnet/decode_it5

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 10 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y17_nnet/decode_it10

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --iter 15 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4y17_nnet/decode_it15
)


(  # 4y18 is as 4y16, but using half the samples per iter, and
   # double the #iters.  The WER is about the same but confusingly,
   # the validation objf is much worse for 4y18, -1.51 vs -1.43.
   # This seems to be related to the fact that 4y18 has much lower
   # learning rates at the end.

  steps/train_nnet_cpu_parallel5.sh --num-iters 30  --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
   --samples-per-iteration 100000 \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y18_nnet
 for iter in 10 20 30; do
   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --iter $iter \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y18_nnet/decode_it$iter
 done
)


(  # 4y19 is as 4y18, but changing overshoot from 1.8 (default) to 1.5
  # Note: validation objf is much better than 4y18 (-1.44 vs -1.51) but WER 
  # is very slightly worse.  Not sure what to make of it.
  steps/train_nnet_cpu_parallel5.sh --num-iters 30  --alpha 4.0 --nnet-config-opts "--learning-rate 0.0025" \
    --overshoot 1.5 \
     --samples-per-iteration 100000 \
    --minibatch-size 1000 --minibatches-per-phase 200 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y19_nnet
 for iter in 10 20 30; do
   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --iter $iter \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y19_nnet/decode_it$iter
 done
)


( # 4y20 is the "parallel6.sh" training script, which has learning
  # rates varying according to a schedule fixed in advance.
  steps/train_nnet_cpu_parallel6.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.0025 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y20_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y20_nnet/decode

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode --iter 30 \
     exp/tri3b/graph data/test exp/tri4y20_nnet/decode_it30

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 --config conf/decode.config \
      --transform-dir exp/tri3b/decode --iter 30 \
      exp/tri3b/graph data/test exp/tri4y20_nnet/decode_it30_wide &
)


( # 4y21 is as 4y20 but double the final learning rate.
  steps/train_nnet_cpu_parallel6.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.0025 --final-learning-rate 0.0005 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y21_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y21_nnet/decode
)

( # 4y22 is as 4y20 but double the initial *and* final learning rate,
  # cf 21 which is double final only.
  steps/train_nnet_cpu_parallel6.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.005 --final-learning-rate 0.0005 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y22_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y22_nnet/decode
)


( # 4y23 is related to 4y{20,21,22}-- in this case we have double initial
  # learning rate but the same final one, so it decreases faster than before.
  # CAUTION: code was broken!  Only 1st model was taken.
  steps/train_nnet_cpu_parallel6.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.005 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y23_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y23_nnet/decode
)


( # 4y24 is the most similar to 4y23, but again doubling the initial learning rate.
  # CAUTION: code was broken!  Only 1st model was taken.
  steps/train_nnet_cpu_parallel6.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.01 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y24_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y24_nnet/decode
)


( # 4y25 is as 4y24, but with shrinking (the default in the 7 script)
  # CAUTION: code was broken!  Only 1st model was taken.
  steps/train_nnet_cpu_parallel7.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.01 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y25_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y25_nnet/decode
)

( # 4y26 is as 4y25, but with fewer iterations-- seeing if we can get the
  # same performance quicker.  Note: it's a bit worse (2.19 vs 1.97).
  # CAUTION: code was broken!  Only 1st model was taken.
  steps/train_nnet_cpu_parallel7.sh --num-iters 20  --alpha 4.0 \
    --initial-learning-rate 0.01 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y26_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y26_nnet/decode
)

( # 4y27 is as 4y26, but with even higher initial learning rate.
  #[ Note: 4y26 was with broken code, so don't compare!]
  steps/train_nnet_cpu_parallel7.sh --num-iters 20  --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y27_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y27_nnet/decode
)

( # 4y28 is as 4y25, but using the 8 script which does nnet-combine
 # (without the previous nnet) instead of averaging and then shrinking.
  steps/train_nnet_cpu_parallel8.sh --num-iters 30  --alpha 4.0 \
    --initial-learning-rate 0.01 --final-learning-rate 0.00025 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y28_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y28_nnet/decode
)

( # 4y29 is as 4y27, but with a 10x higher final learning rate.
  # NOTE: it's taking about as long to do the "shrinking" each time
  # than it is to do the training, so we need to make the validation
  # set smaller.
  steps/train_nnet_cpu_parallel7.sh --num-iters 20  --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.002 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y29_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y29_nnet/decode
)

( # 4y30 is as 4y29, but with even higher final learning rate.
  # *after* the "combine" stage, the validation objf is *very* slightly better
  # than 4y29, but the WER is non-significantly worse.
  steps/train_nnet_cpu_parallel7.sh --num-iters 20  --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y30_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y30_nnet/decode
)

( # 4y31 is as 4y30, but fewer iters (15).  WER is worse (1.85->2.03).
  steps/train_nnet_cpu_parallel7.sh --num-iters 15  --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y31_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y31_nnet/decode
)

( # 4y32 is as 4y30, but using the "8" script which does "combine" instead
  # of average and then shrink.  WER is a bit worse, and also validation set
  # performance is slightly worse.
  steps/train_nnet_cpu_parallel8.sh --num-iters 20  --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y32_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y32_nnet/decode
)


( # 4y33 is as 4y30, but more iterations (30 not 20).
  # It's worse, not better.  And the objf is worse, 1.25 vs 1.22.
  steps/train_nnet_cpu_parallel7.sh --num-iters 30 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y33_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y33_nnet/decode
)

( # 4y34 is as 4y30, but using the "9" version of the script which
  # is faster in clock time because it doesn't usually wait for the
  # "shrink" process.
  # (Good-- it gives basically the same results: 1.84 vs 1.85.)
  steps/train_nnet_cpu_parallel9.sh --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y34_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y34_nnet/decode
)


( # 4y35 is as 4y34, but going from the "9" to "10" script; this
  # abandons the way of shrinking using the previous iter's shrinkage
  # parameters as it seemed to cause a strange instability.  Instead
  # we compute the shrinkage parameters every 3 iters by default, in the
  # normal way, and for the intermediate iters we use the last iter's 
  # parameters.  Also we use fewer samples (2k) to do the shrinking.

  # The WER is a bit worse, 1.84 -> 1.97%.  Also the validation-set
  # performance is a bit worse, although not measured on the exact same
  # frames: -1.225 -> -1.238. 
  # (same happened in SWBD; will try --add-layers-period 1).

  steps/train_nnet_cpu_parallel10.sh --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y35_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y35_nnet/decode
)


( # 4y36 is as 4y35 but trying to tune alpha again; going from 4.0 to 2.0.
  # WER was fractionally worse (0.01%), but validation objf was worse by 0.02.
  steps/train_nnet_cpu_parallel10.sh --num-iters 20 --alpha 2.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y36_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y36_nnet/decode
)


( # 4y37 is as 4y35, but using the "11" script which re-builds the
  # tree; for now, using the same #leaves as the baseline (1800).
  # Note: performance is about the same, as expected.
  steps/train_nnet_cpu_parallel11.sh --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y37_nnet 1800

   utils/mkgraph.sh data/lang exp/tri4y37_nnet exp/tri4y37_nnet/graph || exit 1;

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri4y37_nnet/graph data/test exp/tri4y37_nnet/decode
)


( # 4y37b is as 4y37 but using more leaves: 3k vs 1800.
  # Didn't help, WER 1.93 -> 1.95
  steps/train_nnet_cpu_parallel11.sh --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y37b_nnet 3000

   utils/mkgraph.sh data/lang exp/tri4y37b_nnet exp/tri4y37b_nnet/graph || exit 1;

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri4y37b_nnet/graph data/test exp/tri4y37b_nnet/decode
)


( # 4y38 is as 4y37 but using a two-level tree ("12" script), which we will
  # later use (probably after modifying the "12" script), together with mixture weights in the model.
  # (Yes, it works; actually it's better, 1.83 vs 1.93.  Probably this is becaudse
  # is has more leaves; 1648 vs 1487.)
  steps/train_nnet_cpu_parallel12.sh --num-iters 20 --alpha 4.0 \
    --use-mixtures false \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y38_nnet 500 1800

   utils/mkgraph.sh data/lang exp/tri4y38_nnet exp/tri4y38_nnet/graph || exit 1;

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri4y38_nnet/graph data/test exp/tri4y38_nnet/decode
)

( # 4y38b is running 4y38 again after modifying the scripts to add
  # the MixtureProbComponent.  I'm making this done by options,
  # so we can replicate the old results.
  steps/train_nnet_cpu_parallel12.sh --num-iters 20 --alpha 4.0 \
    --use-mixtures false \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y38b_nnet 500 1800

   utils/mkgraph.sh data/lang exp/tri4y38b_nnet exp/tri4y38b_nnet/graph || exit 1;

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri4y38b_nnet/graph data/test exp/tri4y38b_nnet/decode
)



( # 4y39 is as 4y35 (10.sh script), but --add-layers-period 1.
  # helped slightly 1.97 -> 1.91, and objf slightly better -1.257 -> -1.252
  steps/train_nnet_cpu_parallel10.sh --num-iters 20 --alpha 4.0 \
    --add-layers-period 1 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y39_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y39_nnet/decode
)

( # 4y40 is as 4y35 (or 4y39) but --initial-num-hidden-layers 2,
  # so we add both hidden layers at the start.
  # WER better 1.97->1.87, and objf slightly better -1.257 -> -1.250
  steps/train_nnet_cpu_parallel10.sh --num-iters 20 --alpha 4.0 \
    --initial-num-hidden-layers 2 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y40_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y40_nnet/decode
)


( # 4y41 is as 4y35 but using dropout-- rate of 0.5. (with the 13 script)
  # The learning seemed to be noisy or unstable -> halved the initial learning rate.
  steps/train_nnet_cpu_parallel13.sh --dropout-proportion 0.5 --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.01 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y41_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y41_nnet/decode
)


( # 4y42 is as 4y35 (also c.f. 4y41), but using additive noise with a standard
  # deviation of 0.2.  Went back to original learning rate of 4y35.
  steps/train_nnet_cpu_parallel13b.sh --additive-noise-stddev 0.2 --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y42_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y42_nnet/decode
)

( # 4y43 is as 4y42 but using a larger standard deviation, 0.5 rather than 0.2.
  steps/train_nnet_cpu_parallel13b.sh --additive-noise-stddev 0.5 --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y43_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y43_nnet/decode
)

( # 4y44 is as 4y43 but doubling the #parameters.
  # c.f. 4y44b which is the real baseline.  The additive noise does not help:
  # this (1.95 WER) is worse than 4y44b (1.99 WER).  Also validation performance
  # is worse here (-1.210 vs  -1.207).
  steps/train_nnet_cpu_parallel13b.sh --additive-noise-stddev 0.5 --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 2000000  data/train data/lang exp/tri3b_ali exp/tri4y44_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y44_nnet/decode
)

( # 4y44b is a baseline for 4y44 with no additive noise.
  steps/train_nnet_cpu_parallel13b.sh --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 2000000  data/train data/lang exp/tri3b_ali exp/tri4y44b_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y44b_nnet/decode
)

( # 4y45 is as 4y35, but using --local-balance false.
  # This is needed for speed on larger setups; here, I just want to
  # see if it affects the WER.
  # WER almost the same, 1.97 -> 1.94.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 20 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 100000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45_nnet/decode

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode --iter 20 \
     exp/tri3b/graph data/test exp/tri4y45_nnet/decode_it20 
)


( # 4y45b is as 4y45, but using double the #iters and add-layers-period and
  # shrink-interval, and half the samples per iter.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 40 --add-layers-period 4 \
    --shrink-interval 6 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 50000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45b_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45b_nnet/decode
)

( # 4y45b2 is as 4y45b, but reverting shrink-interval to 3. [might have been causing problems.]
  # and also removing realign-iters.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 40 --add-layers-period 4 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 50000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45b2_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45b2_nnet/decode
)


( # 4y45c is as 4y45, but using half the #iters and add-layers-period and
  # shrink-interval, and double the samples per iter.  Also c.f. 4y45b.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 10 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 200000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45c_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45c_nnet/decode
)


( # 4y45c3 is as 4y45c, but using twice the initial learning rate.
  # (compare also with 4y45d3)
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 10 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.04 --final-learning-rate 0.004 \
    --samples-per-iteration 200000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45c3_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45c3_nnet/decode
)


( # 4y45d is as 4y45, but using 4x the #iters and add-layers-period and
  # shrink-interval, and 1/4 the samples per iter.
  # This is kind of broken-- there was an instability RE shrinking.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 80 --add-layers-period 8 \
    --shrink-interval 12 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 25000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45d_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45d_nnet/decode
)

( # 4y45d2 is as 4y45d, but with shrink-interval 3, as it was somehow unstable; also
  # removing --realign-iters.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 80 --add-layers-period 8 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 25000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45d2_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45d2_nnet/decode
)


( # 4y45d3 is as 4y45d2, but with double the initial learning rate.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 80 --add-layers-period 8 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.04 --final-learning-rate 0.004 \
    --samples-per-iteration 25000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45d3_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45d3_nnet/decode
)

( # 4y45d4 is as 4y45d, but using the 10b script and --apply-shrinking false,
  # so it doesn't apply the shrinking between iters.  Also not realigning.
  # Interesting!! WER goes from 2.05% -> 1.85%, similar to 4y45e.

  steps/train_nnet_cpu_parallel10b.sh --apply-shrinking false \
    --local-balance false --num-iters 80 --add-layers-period 8 \
    --shrink-interval 12 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 25000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45d4_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45d4_nnet/decode
)

( # 4y45d5 is as 4y45d, but --shrink-interval 1.  Trying to see why
  # 4y45d5e was better than 4y45d5.
  # Interesting!  It is worse than 4y45d4, 1.85 -> 2.01.
  # Objf also a bit worse, -1.24 -> -1.25.

  steps/train_nnet_cpu_parallel10.sh \
    --local-balance false --num-iters 80 --add-layers-period 8 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 25000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45d5_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45d5_nnet/decode
)


( # 4y45e is as 4y45, but using around 1/4 the #iters and add-layers-period and
  # shrink-interval, and 4x the samples per iter.  Also c.f. 4y45c (which has
  # double the #samples per iter
  # Note: we actually use 6 iters not 5, because the after the first iter we add a layer. 
  # This is really weird.  The WER performance is *better* than the baseline 4y45,
  # 1.94 -> 1.86.  However, objf is worse, -1.24 -> -1.26.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 6 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45e_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45e_nnet/decode

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     --iter 6 exp/tri3b/graph data/test exp/tri4y45e_nnet/decode_it6
)


( # 4y45e2 is as 4y45e (400k samples per iter), but using more iters, 6->10.
 # Note: WER is better than 4y45e, 1.86->1.84, and
 # validation objf is much better, -1.26 -> -1.20.
 # This seems very promising.
 # CAUTION: I seem to have overwritten this directory below.
 # Renaming it to 4y45e2old, and rerunning.

  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 10 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45e2old_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45e2old_nnet/decode
)


( # 4y45e2 is as 4y45e but shrink=false.  [ worse, 1.86 -> 1.99% WER]
  # CAUTION: I seem to have overwritten a directory of the same name above.
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 6 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink false --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45e2_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45e2_nnet/decode
)

( # 4y45e4 is as 4y45e but with shrink-interval=2.
  # Still trying to find out how frequently we should
  # "shrink"-- seems to be helpful but not if done too
  # often.
  # (it's almost the same as 4y45e, in both WER (1.86->1.88), and
  # validation objf).
  steps/train_nnet_cpu_parallel10.sh --local-balance false --num-iters 6 --add-layers-period 1 \
    --num-iters-final 5 \
    --shrink-interval 2 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y45e4_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y45e4_nnet/decode
)



( 
  # 4y46 is as 4y45e2, but the *old* version (see above) where I was using 10 iters
  # (WER was 1.84%, validation objf was -1.20).  I am adding the "mixing up" to it.
  # Now, 1800 was the target #leaves in the original system (tri3b), so I'm using
  # 4000 as a reasonable number to mix up to.
  # using the 10c script which supports the --mix-up option.
  # It is better than the baseline, 1.84% -> 1.78%, -1.20 -> -1.16.

  steps/train_nnet_cpu_parallel10c.sh --local-balance false --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y46_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y46_nnet/decode
)


( 
  # 4y47 is as 4y46 but redoing it after changing the code.
  # Caution: I temporarily changed the script to say Path not path, so I could
  # use a different binary directory.
  # (almost the same: 1.78 -> 1.75% WER)
  steps/train_nnet_cpu_parallel10c.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47_nnet/decode
)


( 
  # 4y47b is as 4y47 but redoing it after changing the code so that 
  # the bias is no longer handled in a "special" way but is just an extra dimension of 1.0.
  #  Note: should actually compare with 4y48 because I had to change the 10c
  # script to be as 10d before it would run with current code.
  steps/train_nnet_cpu_parallel10c.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47b_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47b_nnet/decode
)

( 
  # 4y47b2 is just a rerun of 4y47b, after rewriting some code; it should
  # be the same as before but just checking.
  # (yes, it's the same essentially, only 0.02% difference.)
  steps/train_nnet_cpu_parallel10c.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47b2_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47b2_nnet/decode
)

( 
  # 4y47c is as 4y47b but using the 10d script which is otherwise now identical
  # to the 10c script, and adding the option --l2-factor 2.0 which we multiply by the
  # inverse #frames to get the l2 penalty.  This then get applied by the
  # AffineComponentPreconditioned code.

  steps/train_nnet_cpu_parallel10d.sh --num-iters 10 --add-layers-period 1 \
    --l2-factor 2.0 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47c_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47c_nnet/decode
)

( 
  # 4y47d is as 4y47c but changing l2-factor from 2.0 to 10.0
  steps/train_nnet_cpu_parallel10d.sh --num-iters 10 --add-layers-period 1 \
    --l2-factor 10.0 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47d_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47d_nnet/decode
)

( 
  # 4y47e is as 4y47d but changing l2-factor from 10.0 to 50.0
  # WER gets a bit worse, 1.80 -> 1.85.  Doesn't even seem to changes
  # the shrinkage factors much.
  steps/train_nnet_cpu_parallel10d.sh --num-iters 10 --add-layers-period 1 \
    --l2-factor 50.0 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47e_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47e_nnet/decode
)

( 
  # 4y47f is as 4y47e but re-running after fixing the code that was not 
  # calling srand as it should have been.  (All previous expts were broken.)
  # [note: not clearly better.]
  steps/train_nnet_cpu_parallel10d.sh --num-iters 10 --add-layers-period 1 \
    --l2-factor 50.0 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y47f_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y47f_nnet/decode
)


( 
  # 4y48 is as 4y47 but after another code change to use posteriors not
  # alignments in the nnet-egs code.
  # WER slightly worse (1.75 -> 1.80%) but probably random.
  steps/train_nnet_cpu_parallel10d.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y48_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y48_nnet/decode
)

( 
  # 4y49 is as 4y48 but using the 10e script which calls nnet-train-parallel,
  # which uses multi-threaded and hogwild.
  # Had to decrease learning rate, as I found instability.
  # This was worse (1.80 -> 2.00), which was not unexpected due to the
  # lower learning rates I had to use, and the objf was worse on validation set (-1.17 -> -1.27).
  steps/train_nnet_cpu_parallel10e.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 1 --alpha 4.0 \
    --initial-learning-rate 0.005 --final-learning-rate 0.002 \
    --samples-per-iteration 400000 --minibatch-size 1000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y49_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y49_nnet/decode
)


( 
  # 4y50 is as 4y49 (also c.f. 4y48) but using the 10f script which uses nnet-combine-fast
  # for speed, does not apply shrinkage except every "shrink-interval" iters,
  # and uses by default a smaller minibatch size of 128.  Going back to the
  # same learning rates as 4y48.

  steps/train_nnet_cpu_parallel10f.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y50_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y50_nnet/decode
)


(
  # 4y51 is as 4y50 but using the 10g script where we have streamlined the "randomize"
  # process.

  steps/train_nnet_cpu_parallel10g.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y51_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y51_nnet/decode
)

(
  # 4y52 it as 4y51 but with further streamlining of how the "randomizing" process
  # is done to avoid latency.
  # (Even better WER, 1.71->1.68, but surely random.)

  steps/train_nnet_cpu_parallel10h.sh --num-iters 10 --add-layers-period 1 \
    --mix-up 4000 \
    --num-iters-final 5 \
    --shrink-interval 3 --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --samples-per-iteration 400000 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4y52_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4y52_nnet/decode
)



( 
  # 4z is the first experiment with the batch update-- we
  # first do 3 iters of SGD and then start batch stuff.
  # which uses multi-threaded and hogwild.
  # Had to decrease learning rate, as I found instability.
  steps/train_nnet_cpu_batch.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z_nnet/decode
)

( 
  # 4z1 is as 4z after a small script fix (remove --scale for preconditioned model)
  # (copied the whole dir and starting from --stage 5).
  steps/train_nnet_cpu_batch.sh --add-layers-period 1 \
    --stage 5 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z1_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z1_nnet/decode
)&

( 
  # 4z2 is as 4z1 but changing --precon-alpha from 0.1 to 1.0.
  steps/train_nnet_cpu_batch.sh --add-layers-period 1 \
    --stage 5 \
    --mix-up 4000 \
    --alpha 4.0 \
    --precon-alpha 1.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z2_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z2_nnet/decode
)&

( 
  # 4z3 is as 4z2 but changing --precon-alpha from 0.1 to 0.01
  steps/train_nnet_cpu_batch.sh --add-layers-period 1 \
    --stage 5 \
    --mix-up 4000 \
    --alpha 4.0 \
    --precon-alpha 0.01 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z3_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z3_nnet/decode
)&


( 
  # 4z4 is as 4z1, but using the batch2.sh script which only uses, by default,
  # half the data on each iteration, but does more iterations (30 vs 20).
  steps/train_nnet_cpu_batch2.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z4_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z4_nnet/decode
)&

( 
  # 4z5 is as 4z4, but using half validation frames and half training
  # frames in the validation set.
  steps/train_nnet_cpu_batch2.sh --add-layers-period 1 \
    --num-valid-frames 2000 --num-train-frames-in-valid 2000 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z5_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z5_nnet/decode
)&


( 
  # 4z6 is as 4z4, but using training subset not real 
  # validation subset, until close to the end (batch3 script)

  steps/train_nnet_cpu_batch3.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z6_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z6_nnet/decode
)&


( 
  # 4z7 is as 4z5, but using entirely training-data subset
  # for a "validation set".
  steps/train_nnet_cpu_batch2.sh --add-layers-period 1 \
    --num-valid-frames 0000 --num-train-frames-in-valid 4000 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z7_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z7_nnet/decode
)&


( 
  # 4z8 is as 4z5, but using 1/4 of training-data subset and 
  # 3/4 validation.
  steps/train_nnet_cpu_batch2.sh --add-layers-period 1 \
    --num-valid-frames 3000 --num-train-frames-in-valid 1000 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z8_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z8_nnet/decode
)&



( 
  # 4z9 is using the batch4 script which combines with the
  # training set but uses the validation set for shrinking.

  steps/train_nnet_cpu_batch4.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z9_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z9_nnet/decode
)&



( 
  # 4z10 is using the batch5 script which uses different subsets
  # of the validation data each time.  Compare with 4z5 which also
  # uses half-and-half validation and training.
  # Results very similar to 4z5 (1.68 vs 1.60)... not surprising.

  steps/train_nnet_cpu_batch5.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z10_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z10_nnet/decode
)&


( 
  # 4z11 is as as 4z9, but using the "batch6" script which
  # is as the batch4 script but adding nnet-am-fix to deal with
  # sigmoids that are in the flat region by decreasing the corresponding
  # parameters.  
  # Note: the WER gets a bit worse (1.58 -> 1.78), and the validation performance
  # of the last model is worse ( -1.338 -> -1.364), and the validation performance of
  # the final combined model is slightly worse (-1.2911 -> -1.2968).
  # It seems that this setup "wanted" fewer parameters, and by fixing the neural net
  # this way, we actually gave it more parameters, which made the performance a bit
  # worse.

  steps/train_nnet_cpu_batch6.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z11_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z11_nnet/decode
) &

( 
  # 4z11b is as 4z11, but using fewer parameters, 750K instead of 1M.
  # Slight improvement, 1.78 -> 1.75.
  steps/train_nnet_cpu_batch6.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 750000  data/train data/lang exp/tri3b_ali exp/tri4z11b_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z11b_nnet/decode
) &


( 
  # 4z12 is as 4z10, but doing from batch5 to batch7 script, which
  # adds the fix-nnet stuff.  Also compare with the 9 -> 11 experiments.
  # this differs from 4z11 only slightly, in that it uses different
  # subsets of validation data each time.
  #  (slightly worse than 4z10, probably because of more overtraining, and
  # slightly but probably non-significantly better than 4z11.)

  steps/train_nnet_cpu_batch7.sh --add-layers-period 1 \
    --mix-up 4000 \
    --alpha 4.0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.01 \
    --samples-per-iteration 400000 --minibatch-size 1024 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "20" \
    --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4z12_nnet

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4z12_nnet/decode
)&

## starting again in trunk; based on exp/tri4y52_nnet/ in old setup.
 # a1 is the first experiment, 
( steps/train_nnet_cpu.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 --num-iters-final 5 \
     --shrink-interval 3 --alpha 4.0 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --num-parameters 1000000 \
      data/train data/lang exp/tri3b_ali exp/tri4a1_nnet 

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a1_nnet/decode )

# a2 is as a1, but fewer parameters (1M->750K)
(   steps/train_nnet_cpu.sh  --num-epochs 20 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 --num-iters-final 5 \
     --shrink-interval 3 --alpha 4.0 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --num-parameters 750000 \
      data/train data/lang exp/tri3b_ali exp/tri4a2_nnet 

   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/tri4a2_nnet/decode ) 
