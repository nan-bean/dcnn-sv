#Torch DCNN
* The implementation is based on [Wespeaker](https://github.com/wenet-e2e/wespeaker"Wespeaker") repository. We offer an evaluation script for reproducing.

##Model config
* Feature: 80-dim fbank, online data augmentation (additive noise, reverberation, speed perturb), score normalization
* Metrics: EER(%), MinDCF(P_tar=0.01)

##Voxceleb Results
Train set: Voxceleb2-dev, 5994 speakers
Test set: Voxceleb-O

|__Model__|__Vox-O__  |__Vox-E__  |__Vox-H__  |
|DyKCNN   |0.712/0.068|0.952/0.111|1.780/0.184|
|+asnorm  |0.617/0.056|0.915/0.102|1.680/0.167|



