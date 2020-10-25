# technica_hackathon

Used linear regression, svm and randomforest to predict survivor on titanic

data from:https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html


## Inspiration
Titanic is one of the most serious and famous shipwreck accidents in history. Having a list of 800 and more passenger's list, machine learning techniques are used to find what is the most important features that leads to a passenger's survival. The list is obtained on the following website (https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html). Moreover, the models can be used to predict if a passenger is going to survive the shipwreck or not. This can help improve survival rates of passages regardless their age, ship fare they paid or sex, etc. 

## What it does
The machine learning models can predict a passenger is going to survive the shipwreck or not with high accuracy. They also identify which are the most important features that determine if an individual can survive or not.

## How we built it
A python library, sklearn, is used to build those machine learning models. In this project, 3 models have been built - linear regression, random forest, and SVM. They are also evaluated using ROC curves to ensure the models are running with high accuracy. 20% of the data are reserved for testing use. 

## Challenges we ran into
We tried to build a model using google AutoML. However, the sample size is too small as AutoML requires 1000 or above samples for the model to be accurate. The models we built using sklearn ran fine.

## Accomplishments that we're proud of
The models we built are with high accuracy and we found out that how sex determines the most if a passenger survives or not. 
Accuracy:
Linear regression: 80.3%

## What we learned
We learned that females survive much better in an accident in the past because men took the responsibility to save women so much as they thought that women were weaker in general. We found that discrimination against sex in the past did not have to favor men all the time. In this case, more women survived.

## What's next for Titanic
Different machine learning models can be used to predict the survival of passengers. Moreover, sex can be eliminated to see what is the next most important feature is that affects the survival rate. If more information about the passengers is obtained, we can also explore other factors that affect the survival rate of passengers. 
