# ReSpeakerExperiments
Trying new things with Respeaker

DOA file: 
- Prints direction of most significant sound (terms of degree)
- Prints whether the main source of sound is voice or not 

Record file:
- Set clip is recorded and output to a .wav file 

Noise classification: 
- Ambient noise: Urban sound 8K dataset. Background and foreground classification
- Crowd noise: Multi-speaker corpus

# Prerequisites
a. This predictive model uses tensorflow and keras via python. Please familiarise yourself with the theory before using this repo

b. Install requirements

# Steps:
1. Download git repo
2. Create DataForReSpeaker folder - move 'allData.csv' and 'equalDataFolds.csv' into folder
3. Create 'allAudio' folder within DataForReSpeaker folder
4. Download .wav files from https://www.kaggle.com/datasets/katb21/soundclassification into 'allAudio'
5. Run process_data.py
6. Run create_model.py
7. Run test_model.py
8. Attach the respeaker to your device - using USB. A green light should appear on the ReSpeaker if connected correctly
9. Run get_index.py to know which port to use to access the ReSpeaker
10. Change RESPEAKER_INDEX variable with input Device id in live_classification.py 
11. Run live_classification.py

Example:
``` 
*get_index output*
Input Device id  0  -  Built-in Microphone
Input Device id  2  -  ReSpeaker 4 Mic Array (UAC1.0)

*live_classification input*
RESPEAKER_INDEX = 2
```

Output from live_classification:
Direction, Noise-classification
