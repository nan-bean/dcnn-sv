#!/bin/bash
# Copyright      2018   Hossein Zeinali (Brno University of Technology)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.
#
# Pretrained models are available for this recipe.
# See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

train_cmd=
nnet_dir=exp/xvector_tf_dcnn
stage=0
train_stage=-1
iter=

. ./cmd.sh
. ./path.sh
set -e
. ./utils/parse_options.sh

mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
vaddir=`pwd`/mfcc

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue

if [ ${stage} -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/mnt/matylda2/data

  # Prepare telephone and microphone speech from Mixer6.
  #local/make_mx6.sh ${data_root}/LDC2013S03 data/
  local/make_mx6_BUT.sh ${data_root}/LDC/LDC2013S03_Mixer6-Speech data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  #local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/
  local/make_sre10.pl ${data_root}/NIST/sre10/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  #local/make_sre08.pl ${data_root}/LDC2011S08 ${data_root}/LDC2011S05 data/
  local/make_sre08_BUT.pl ${data_root}/NIST/sre08/download ${data_root}/NIST/sre08/sp08-11/test ${data_root}/NIST/sre08/sp08-11/train data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre_BUT.sh ${data_root} data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1  \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1_BUT.pl ${data_root}/SWITCHBOARD/sw_cellular_part1 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2_BUT.pl ${data_root}/SWITCHBOARD/sw_cellular_part2 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1_BUT.pl ${data_root}/SWITCHBOARD/sw2_phase1 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2_BUT.pl ${data_root}/SWITCHBOARD/sw2_phase2 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3_BUT.pl ${data_root}/SWITCHBOARD/sw2_phase3 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval_BUT.pl ${data_root}/NIST/sre16/R149_0_1 data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl ${data_root}/NIST/sre16/LDC2016E46_SRE16_Call_My_Net_Training_Data data

fi

if [ ${stage} -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_major; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "${train_cmd}" \
      data/${name} exp/make_mfcc ${mfccdir}
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "${train_cmd}" \
      data/${name} exp/make_vad ${vaddir}
    utils/fix_data_dir.sh data/${name}
    
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_fbank ${fbankdir}
    utils/fix_data_dir.sh data/${name}
  done

  utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre
  utils/fix_data_dir.sh data/swbd_sre
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ ${stage} -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=${frame_shift} '{print $1, $2*frame_shift;}' \
    data/swbd_sre/utt2num_frames > data/swbd_sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/swbd_sre data/swbd_sre_reverb
  cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
  rm -rf data/swbd_sre_reverb
  mv data/swbd_sre_reverb.new data/swbd_sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /mnt/matylda2/data/MUSAN/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "10:5" --fg-noise-dir \
    "data/musan_noise" data/swbd_sre data/swbd_sre_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "10:7:5" --num-bg-noises "1" \
    --bg-noise-dir "data/musan_music" data/swbd_sre data/swbd_sre_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "19:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/musan_speech" data/swbd_sre data/swbd_sre_babble

  # Combine comp, reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/swbd_sre_aug data/swbd_sre_reverb data/swbd_sre_noise \
    data/swbd_sre_music data/swbd_sre_babble

  utils/fix_data_dir.sh data/swbd_sre_aug

  # Make filterbanks for the augmented data. Note that we do not compute a new
  # vad.scp file here. Instead, we use the vad.scp from the clean version of
  # the list.
    
  steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "${train_cmd} --long 1" \
    data/swbd_sre_aug exp/make_fbank ${fbankdir}

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug data/swbd_sre

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
  utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  utils/fix_data_dir.sh data/sre_combined
fi

# Now we prepare the features to generate examples for xvector training.
if [ ${stage} -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "${train_cmd}" \
    data/swbd_sre_combined data/swbd_sre_combined_no_sil exp/swbd_sre_combined_no_sil
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_combined_no_sil/utt2num_frames.bak > data/swbd_sre_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2spk > data/swbd_sre_combined_no_sil/utt2spk.new
  mv data/swbd_sre_combined_no_sil/utt2spk.new data/swbd_sre_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_combined_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
  mv data/swbd_sre_combined_no_sil/spk2utt.new data/swbd_sre_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2spk data/swbd_sre_combined_no_sil/utt2num_frames > data/swbd_sre_combined_no_sil/utt2num_frames.new
  mv data/swbd_sre_combined_no_sil/utt2num_frames.new data/swbd_sre_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil
fi

local/tf/run_xvector.sh --stage ${stage} --train-stage ${train_stage} \
  --data data/swbd_sre_combined_no_sil --nnet-dir ${nnet_dir} \
  --egs-dir ${nnet_dir}/egs

if [ ${stage} -le 7 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  local/tf/extract_xvectors.sh --cmd "${train_cmd} --mem 6G" --nj 40 \
    ${nnet_dir} data/sre16_major \
    ${nnet_dir}/xvectors_sre16_major

  # Extract xvectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  local/tf/extract_xvectors.sh --cmd "${train_cmd} --long 1 --mem 12G" --nj 35 \
    ${nnet_dir} data/sre_combined \
    ${nnet_dir}/xvectors_sre_combined

  # The SRE16 test data
  local/tf/extract_xvectors.sh --cmd "${train_cmd} --mem 6G" --nj 40 \
    ${nnet_dir} data/sre16_eval_test \
    ${nnet_dir}/xvectors_sre16_eval_test

  # The SRE16 enroll data
  local/tf/extract_xvectors.sh --cmd "${train_cmd} --mem 6G" --nj 40 \
    ${nnet_dir} data/sre16_eval_enroll \
    ${nnet_dir}/xvectors_sre16_eval_enroll
fi

if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  ${train_cmd} ${nnet_dir}/xvectors_sre16_major/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/xvectors_sre16_major/xvector.scp \
    ${nnet_dir}/xvectors_sre16_major/mean.vec || exit 1;

  lda_dim=100
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  ${train_cmd} ${nnet_dir}/xvectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre_combined/xvector.scp ark:- |" \
    ark:data/sre_combined/utt2spk ${nnet_dir}/xvectors_sre_combined/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/xvectors_sre_combined/log/plda.log \
    ivector-compute-plda ark:data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre_combined/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/xvectors_sre_combined/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  ${train_cmd} ${nnet_dir}/xvectors_sre16_major/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    ${nnet_dir}/xvectors_sre_combined/plda \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre16_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${nnet_dir}/xvectors_sre16_major/plda_adapt || exit 1;
fi

if [ ${stage} -le 9 ]; then
  # Get results using the out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/sre16_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre_combined/plda - |" \
    "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec scp:${nnet_dir}/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre16_eval_scores || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_scores > ${nnet_dir}/scores/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_scores > ${nnet_dir}/scores/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials ${nnet_dir}/scores/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer} %, Tagalog ${tgl_eer} %, Cantonese ${yue_eer} %"
fi

if [ ${stage} -le 10 ]; then
  # Get results using the adapted PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_major/plda_adapt - |" \
    "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec scp:${nnet_dir}/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre16_eval_scores_adapt || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_scores_adapt > ${nnet_dir}/scores/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_scores_adapt > ${nnet_dir}/scores/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials ${nnet_dir}/scores/sre16_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer} %, Tagalog ${tgl_eer} %, Cantonese ${yue_eer} %"
fi

