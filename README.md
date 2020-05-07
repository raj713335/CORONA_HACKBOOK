# CORONA_HACKBOOK

A Deep Learning Algorithm to predict if the chest x-ray scans are of  covid19 positive person or not .



The Training History .
![](Training_history.png)


## Getting Started
- Clone the repo and cd into the directory
```sh
$ git clone https://github.com/raj713335/CORONA_HACKBOOK.git
$ cd CORONA_HACKBOOK
```

- Install tensorflow and all the other required libraries 

```sh
$ pip install tensorflow
```


## Download Dataset From the 
url: https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/download

A X-Ray Report Example .
![](gr1_lrg-a.jpg)


create the test and train dataset , with subfolders of COVID19 and Normal Patients


## Run the coronovirus.py file to train a model and save it as coronovirus.py

```sh
$ coronavirus YT.py
```

## To load and run the model and test it against unknown data\images 


```sh
$ python coronovirus_validate.py
```

