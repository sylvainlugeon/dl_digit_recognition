# Mini Project 1 
EE-559 - Deep Learning - EPFL - 2020

_**Benkley** Tyler_, _**Berdoz** Frédéric_, _**Lugeon** Sylvain_

## Context
The   goal   of   this   project   is   to   assess   the   impact of network  architecture,  weight  sharing  techniques  and  the use  of  auxiliary  losses  in  a  classification  task.  To  do  so, we  compare  the  performance  of  4  architectures (```FCNet```, ```ConvNet```, ```ConvSepNet``` and ```FinalNet```)  subject  to different conditions.


## Content

* ```DL_MP1.ipynb```: Python notebook in which the project was developed. Contains all the figures (and more) presented in the report.

* ```test.py```: Python script that can be run without argument. It will train each one of the 4 models over 200 epochs with an auxiliary loss weighting of f=0.5. **Only one round of training is performed for each model** (the validation over 15 rounds is done in the notebook ```DL_MP1.ipynb```). Its execution time is 322 sec on a dual core 2 GHz _Intel Core i5_.

* ```dlc_practical_prologue.py```: Helper module taken from the moodle page of the course. Imported in both ```DL_MP1.ipynb``` and ```test.py```.

* ```ee559_miniprojects.pdf```: Project definition.

* ```report.pdf```: The report of the project.

* ```data``` folder: Contains the mnist data (or where it will be downloaded the first time).

* ```backup``` folder: Contains backup of variables used for the figures presented in the report.

* ```figures``` folder: Contains figures presented in the report (and others).


## Prerequisite

This code was developped using ```python 3.7.3``` (with its standard libraries), and with ```pytorch 1.4.0```. In addition, for the visualization, ```numpy 1.16.4```, ```matplotlib 3.1.3``` and ```scipy 1.1.0``` were used.





