# Torch DCNN
* The implementation is based on [Wespeaker](https://github.com/wenet-e2e/wespeaker"Wespeaker") repository. Trained model is accessible in the exp/DyKCNN directory.

## Model Config
* Feature: 80-dim fbank, online data augmentation (additive noise, reverberation, speed perturb), score normalization (as-norm)
* Metrics: cosine similarity, EER(%), MinDCF(P_tar=0.01)

## Voxceleb Results
* Train set: Voxceleb2-dev, 5994 speakers
* Test set: Voxceleb-O (Voxceleb-E, Voxceleb-H)

| __Model__ | __Para__    | __Vox-O__   | __Vox-E__   | __Vox-H__   |
|-----------|-------------|-------------|-------------|-------------|
| DyKCNN    | 8.0M        | 0.712/0.068 | 0.952/0.111 | 1.780/0.184 |
| +as-norm  |     -       | 0.617/0.056 | 0.915/0.102 | 1.680/0.167 |




