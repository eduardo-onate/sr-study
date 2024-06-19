# sr-study

This repository contains the code related to my thesis to obtain the degree of Electrical Engineer at the Universidad de Chile, whose translated title is "**Characterization of NVIDIA’s TitaNet-L Model for Speaker Recognition in Spanish**". The document in Spanish can be found in the following link: 


### General Objective
This work aims to characterize NVIDIA's TitaNet-L model in the context of a scalable speaker recognition system in Spanish, analyzing its sensitivity to the number of speakers and to the number and duration of utterances during enrollment and verification.

### Specific Objectives
* Construct a dataset for speaker recognition from labeled sessions from the Chilean Constitutional Process of 2023 (PCCh23) (not shared in this repository).
* Restore the TitaNet-L model's pretrained checkpoint in English.
* Propagate different subsets of data from PCCh23 through TitaNet varying the number and duration of enrollment utterances to study the distribution of vectorial representations in the latent space.
* Design classification rules for the speaker recognition system based on the definition of centroids and cosine similarity.
* Evaluate the performance of the system through the metrics of accuracy, precision, recall, F1, Equal Error Rate (EER) and Minimum Detection Cost Function (MinDCF)
* Provide a replicable methodology for the characterization of biometric systems based on cosine similarity.