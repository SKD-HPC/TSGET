This is the implementation of TSGET: Two-Stage Global Enhanced Transformer for Automatic Radiology Report Generation (https://doi.org/10.1109/JBHI.2024.3350077) at IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024. If you use or extend our work, please cite our paper.

Datasets:

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For IU X-Ray, you can download the dataset from https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing, and then put the files in data/iu_xray.

For MIMIC-CXR, you can download the dataset from https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing, and then put the files in data/mimic_cxr.

Run on IU X-Ray:

Run bash run_iu_xray.sh to train a model on the IU X-Ray data.

Run on MIMIC-CXR:

Run bash run_mimic_cxr.sh to train a model on the MIMIC-CXR data.

Run Reinforcement Learning:

Download the RL-TSGET.rar and prepare the dataset, the evaluation metrics, and the pre-trained models (Under Cross-Entropy Loss)

Run bash run_iu_rl.sh to finetuning the model on the IU X-Ray dataset.

Run bash run_mimic_rl.sh to finetuning the model on the MIMIC-CXR dataset.

Note that: The variance of the experimental results is large when training with Reinforcement Learning on IU X-Ray dataset.
