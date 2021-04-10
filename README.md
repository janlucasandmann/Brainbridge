<img width="280" alt="Brainbridge" src="https://user-images.githubusercontent.com/53909792/114017945-f89f2000-986c-11eb-945c-2e4b889d9531.png">

* * *

Brainbridge tries to build a Brain Machine Interface using single channel EEG. The aim is to build algorithms, that allow reliable communication directly from the brain to any given computer only by thought.

This package includes data collected from single channel EEG measurements from electrodes placed on C4 after the international 10-20 system for EEG.

<img width="300" alt="International 10-20 system for EEG" src="https://user-images.githubusercontent.com/53909792/113484027-975f0180-94a6-11eb-92a8-34b86e820628.png">

All data is collected with a frequency of 100Hz and related to events, in which subjects had to raise their left arm or had to think about raising their left arm or not. Every dataset relates to a different dataset, in which the labels for the different events find themselves in (1 = hand raised, 0 = hand not raised). Each event has a length of 2 seconds and therefore consists of exactly 200 data points. More data is added every week.

* * *

Early studies show a clear correlation between features extracted from the data and the events, which indicates, that event related potential can be identified using the data. Below you can see a complexity matrix for selected features and the events from the extracted data. Correlations between all features and the respective events are marked in the image below, as they represent the most interesting part. The approach is far from perfect, but progresses fastly.

<img width="755" alt="Correlation Matrix" src="https://user-images.githubusercontent.com/53909792/114274014-d9df8b80-9a1c-11eb-8107-b42686057ad5.png">

For any ideas in context of feature selection, machine learning models, measuring methods, or anything else that could help, feel free to reach out!

