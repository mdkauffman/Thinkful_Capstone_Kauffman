# Will it Fund? Predicting Funding Success For Crowdfunded Micro-Loans
## A data science & machine learning project
### by Morgan D. Kauffman


### Introduction
Kiva is a nonprofit organization that offers loans to people in need of philanthropic aid.  Borrowers seek out loans from Kiva for a wide variety of purposes, ranging from small business owners looking for capital infusions to parents seeking help with their children's medical bills or students looking for help paying their tuition.  The loans are funded by a network of microlenders, who donate as little as $25 to loans they wish to support.

Kiva only charges enough interest on the loan to make up for any costs incured by making the loan, such as fees levied by local governments.  Neither Kiva nor the microlenders who fund the loans make any profit off of them; their efforts are entirely philanthropic.  

This is all managed through Kiva's website, where borrowers create a loan profile and lenders choose which profiles (or loans) they want to lend their money to.  It’s very well-designed, upfront about the possibility of people defaulting, and overall appears to be a great philanthropic exercise in ensuring that money gets to the places that need it.

### Purpose

Normally, analysis of a dataset of loans would be focused on default rates.  However, I'd be shocked if Kiva hasn't done that analysis already, and I myself have already done a predictive algorithm for loan default, so I was doubly disinclined to go down that route for this project.  

However, Kiva's business model makes their loan dataset unique, in that the *funding* of a loan is an uncertain variable, alongside whether the borrower will default.  Kiva and its local affiliates extend credit to their borrowers regardless of whether their loan gets funded by Kiva's network of microlenders.  This means that if a loan *fails* to be fully funded, Kiva is left holding the remaining debt and the risk if the borrower should default. 

Because Kiva's operations are funded by charitable donations rather than interest payments, and thus their margins are significantly narrower than a for-profit bank, they would obviously prefer for as much of the risk in their operations to be distributed amongst their crowdfunding network.  Thus, the odds that a loan will be fully funded is almost as important to their business model as the odds that the borrower will default.  

This makes for an interesting deviation from the normal "figure out which loans are more likely to default" predictive analysis that we see in loan datasets, in that whether a loan will be funded is as much a result of emotional or social factors as hard numbers.

### Goal:  
* Prediction of a loan's funding status, based on information available to Kiva at the time the loan was made.

### Data:
The expanded list of Kiva loans, which is available at: https://www.kaggle.com/gaborfodor/additional-kiva-snapshot

### Metrics:
Our two primary metrics will be the overall model success rate and the f1 score for predicting unfunded loans, giving us a sense of how well the models do at pinpointing the loans we're most worried about.

### Methods & Feature Engineering:
Features have been built out of the data that Kiva would have had at the time of the loan being made.  These include:
* The original language of the loan profile
* The coordinates and nearest city/town of the borrower(s)
* The number and gender(s) of the borrowers
* The size and terms of the loan
* The economic sector and general purpose of the loan
* When the profile was posted, and when the money was disbursed to the borrower(s)
* A list of tags that were applied to the loan profile on Kiva's website to help lenders find the profile (though these turned out to have questionable worth, due to uncertainty over when during the funding process they were applied to the loan)
* Various economic indicators for the borrower's nation of residence, including human development index, mean years of schooling, and the like

These features were processed to extract any text-based information, condense the timing of the loan issuance down into two continuous features and a comparison between the posting time and disbursal time, and extracting the number and gender ratio of the borrowers. National economic data and borrower coordinates were used to generate cluster features based on geographic and economic similarity.

### Preliminary Model Comparison: 
Models | Model Score |f1 Score
------------ | ------------- |-------------
Naive Bayes|	0.81|	0.21
Logistic Regression|	0.93|	0.29
KNN|	0.91|	0.35
Decision Tree	|0.94|	0.44
Random Forest	|0.95|	0.50
Gradient Boosting	|0.93|	0.44
Multi-Layer Perceptron	|0.92|	0.44
Keras/Tensorflow	|0.94|	0.25
Convolutional NN	|0.93|	0.27

### Final result of Random Forest, after optimization: 
* Model Score: 0.9552
* Unfunded f1 score: .53

#### Classification Report

  Measure | Precision | Recall | f1 Score  |  Support
------------ | ------------- |------------- |------------- |-------------
class 0: funded |	0.98|	0.97|	.98|	228117
class 1: unfunded |	0.47|	0.59|	.53|	9969
micro avg|	0.96|	0.96|	.96|	238086
macro avg|	0.73|	0.78|	.75|	238086
weighted avg|	0.96|	0.96|	.96|	238086


### Conclusions:

This is, frankly, not a conclusive enough result for Kiva to rely on this model when trying to predict failure-to-fund.  That being said, it still does *much* better than just guessing at random, as we saw with some of the lower-performing algorithms.  

As to why it performs this way, there are two main culprits, based on my analysis. First, while multiple features are faintly but distinctly correlated with our target feature, there are no clear determinative factors that our model can find and use to make its predictions.  Second, even with the faint correlations that it *can* find, many of the loans that are nearly identical in features have different outcomes with regard to funding, significantly muddying the model's ability to clearly identify which loans will be fully funded.

#### Practical Usage:
* Screening for loans at high risk of failure to fund
* Would highlight some problematic loans, though not reliably enough to depend on alone
* Not a replacement for any kind of screening already in use 
* Would need to be used in addition to their normal vetting and any other algorithms they might be employing

#### Known Shortcomings:
* Model performance is not enough to rely on
* NLP-dependent features are left out of the feature-set entirely, likely missing out on a lot of relevant information

#### Further explorations:
* Fully exploring & generating NLP-dependent features
* Optimization of the features that were created via PCA of multi-categorical features
* Deeper analysis of funding rates by time period
* More expansive utilization of clustering
* Integrating localized poverty/economic data from other datasets
