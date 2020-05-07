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

## Download the training and testing dataset from the following url
```sh
$ https://drive.google.com/u/0/uc?export=download&confirm=i2Y7&id=1XjSuZZsFGwH7SpnIPYbw_bRgkP4-ntxw
```


- Install tensorflow and all the other required libraries 

```sh
$ pip install tensorflow
```


## Download Dataset From the 
url: https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/download


## Download The Trained Model From the Following link in case you don't have the computational power to train your model
```sh
url : https://drive.google.com/u/0/uc?id=1112evrjqWlEPw1hkPA44LoVfm_USuys8&export=download
```

# A X-Ray Report Example .

![](gr1_lrg-a.jpg)


## Create the test and train folders , with subfolders of COVID19 and Normal Patients


## Run the coronovirus.py file to train a model and save it as coronovirus.py

```sh
$ coronavirus YT.py
```

## To load and run the model and test it against unknown data\images 


```sh
$ python coronovirus_validate.py
```

