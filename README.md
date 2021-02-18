# Musical Instrument Recognition using Convolutional Neural Networks for NSynth Dataset  

This project aims to study the problem of musical instrument recognition using Convolutional Neural Networks for the NSynth Dataset w.r.t different signal representations of the audio samples.

## Dependencies
The following is the list of libraries/infrastucture used while building and testing this architecture.
* PyTorch 1.7.1
* Librosa 0.8.0
* Matplotlib 3.3.3
* Numpy 1.19.4
* sklearn (For confusion matrix) 0.23.2
* Pandas 1.1.5


## Runtime environment
The following is the runtime environment used while building and testing this architecture.  
CPU: AMD 16 core running Ubuntu 18.04.5 LTS 64GB memory  
GPU: GeForce RTX 2080 Ti 11GB; Driver version: 460.27.04; CUDA version 11.2  

## Project Structure  
* main.py: Main entry point for the project to either train or generate confusion matrix
* /config: Config folder with different config JSON files for each experiment
* /labels: Subset of NSynth labels used for different experiment. One sub-directory for train, test and valid respectively. 
*  /models: Location where models (.pt), loss & accuracy and confusion matrix is saved
* /network: Convolutional Neural Network architecture
* /training: Pytorch code to train the network
* /util: Utilities for plotting etc. 

## Experiments
The following expertiments were conducted as part of this study:  
* Classification using acoustic instruments only
* Classification using combined instrument source-family labels (e.g. electronic-guitar, synthetic-keyboard, acoustic-flute etc.)
* Classification using ensembles (Please see another github project pkamath2/nsynth-cnn-ensemble for this experiment)  

## Running an experiment

Acoustic only instruments - 

1. To train acoustic instruments only, under the config folder, copy the contents of  config-acoustic.json into config.json. Similarly copy the contents of config-combined.json into config.json to run the combined instrument source-family experiments.  
2. Run main.py to kickstart training for the acoustic only -  
```python main.py --operation=train --data_type=mfcc --batch_size=16 --learning_rate=0.0001```
3. data_type currently supports mfcc, mel (for MelSpectrograms), if (for instantaneous frequency) and mel_harmonics (for percussive and harmonics representations in each channel of the input)
