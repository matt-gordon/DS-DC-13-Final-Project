# DS-DC-13-Final-Project
## Activities of Daily Living Classifier based upon wrist-worn accelerometers
#### General Assembly - DC - Data Science Course Final Project
##### by Matthew Gordon    





__Data :__  [Dataset for ADL Recognition with Wrist-worn Accelerometer Data Set](http://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer)   
__Source :__ [University of California - Irvine Machine Learning Repository ](http://archive.ics.uci.edu/ml )    
__Citation :__ Bruno, B., Mastrogiovanni, F., Sgorbissa, A., (2014), Dataset for ADL Recognition with Wrist-worn Accelerometer Data Set, UCI Machine Learning Repository. Irvine,   
Ca: University of California, School of Information and Computer Science   

DESCRIPTION OF DATASET
-------------------------
Human Motion Primitives
--------------------------
The dataset provides labelled recorded executions of a number of simple human activities, which are defined as Human Motion Primitives (HMP):     

| Activity | Description |   
|:---------|--------------|   
|brush_teeth|   to brush one's teeth with a toothbrush|
|climb_stairs|  to climb a number of steps of a staircase|
|comb_hair|    to comb one's hair with a brush|
|descend_stairs| to descend a number of steps of a staircase|
|drink_glass|    to pick a glass from a table, drink and put it back on the table|
|eat_meat|	   to eat something using fork and knife|
|eat_soup|	   to eat something using a spoon (complete gesture)|
|getup_bed|     to get up from a lying position on a bed|
|liedown_bed|    to lie down from a standing position on a bed|
|pour_water|   to pick a bottle from a table, pour its content in a glass on the  table, put it on table|
|sitdown_chair|  to sit down on a chair|
|standup_chair|  to stand up from a chair|
|use_telephone|  to place a telephone call using a fixed telephone|
|walk|	   	to take a number of steps|

Number of recordings per activity
--------------------------------------
| Activity| No. of samples in dataset |
|---------|:------------------------:|
|brush_teeth| 12 |
|climb_stairs | 102 |
|comb_hair | 31 |
|descend_stairs |42 |
|drink_glass |100 |
|eat_meat |5 |
|eat_soup |3 |
|getup_bed |101 |
|liedown_bed |28 |
|pour_water |100 |
|sitdown_chair |100 |
|standup_chair |102 |
|use_telephone| 13 |
|walk |100|

Data Files & Format
--------------------
- zip file, folders organized by activity, tab separated text files of x,y,z sensor output    

- 14 folders, 1460 files, 3 channels (x,y,z ) recorded at 32 samples per second

- Acceleration data recorded in the dataset are coded according to the following mapping:    
      [0; +63] = [-1.5g; +1.5g]

- The conversion rule to extract the real acceleration value from the coded value is the following:   
      real_val = -1.5g + (coded_val/63)*3g


![](/resources/hand.jpg?raw=true)     
[modified from original](https://www.colourbox.dk/preview/2586636-right-hand-isolated-on-the-white.jpg)
