<img width="300" alt="Brainbridge" src="https://user-images.githubusercontent.com/53909792/113483826-aa250680-94a5-11eb-9a85-b2d292ba463c.png">

Brainbridge tries to build a Brain Machine Interface using single channel EEG. The aim is to build algorithms, that allow reliable communication directly from the brain to any given computer only by thought.

This package includes data collected from single channel EEG measurements from electrodes placed on C4 after the international 10-20 system for EEG.

<img width="300" alt="International 10-20 system for EEG" src="https://user-images.githubusercontent.com/53909792/113484027-975f0180-94a6-11eb-92a8-34b86e820628.png">

All data is collected with a frequency of 100Hz and related to events, in which subjects had to raise their left arm or had to think about raising their left arm or not. Every dataset relates to a different dataset, in which the labels for the different events find themselves in (1 = hand raised, 0 = hand not raised). Each event has a length of 2 seconds and therefore consists of exactly 200 data points. More data is added every week.

Early studies show a clear correlation between features extracted from the data and the events, which indicates, that event related potential can be identified using the data. Below you can see a complexity matrix for selected features and the events from the extracted data. Correlations between all features and the respective events are marked in the image below, as they represent the most interesting part. The approach is far from perfect, but progresses fastly.

<img width="755" alt="Bildschirmfoto 2021-04-03 um 17 05 05" src="https://user-images.githubusercontent.com/53909792/113484373-43551c80-94a8-11eb-86c6-232db26cab59.png">

For any ideas in context of feature selection, machine learning models, measuring methods, or anything else that could help, feel free to reach out!

