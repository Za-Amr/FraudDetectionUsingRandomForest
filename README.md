
## Fraud Detection Using Random Forest ##


The aim of this project is to identify the fraudulent credit card transactions. 
For the credit card companies is important to recognize the fraud cases, so that the customers are not charged for items that they did not purchase.

The dataset used for this project contains transactions made by credit cards in September 2013 by european cardholders.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.


Link for the Dataset : https://www.kaggle.com/mlg-ulb/creditcardfraud

Model : we will use the Random Forest and will compare different technique for the unbalanced data

 - Random Forest with Unbalanced Data
 - Random Forest with Oversampling technique using SMOTE
 - Random Forest with Balanced Data
 - Random with Class Weight adjusted manually


Results:
----------

	Model	        Recall	        Precision	     F1	        FN_rate	        FP_rate	        AUCROC
	--------------------------------------------------------------------------------------------------------
	RF_UnBalanced	0.71  	         0.91	        0.80	        28.71	        0.012	        0.974
	--------------------------------------------------------------------------------------------------------
	RF_Smote	0.87    	0.41	        0.56    	12.87           0.23	        0.983
	---------------------------------------------------------------------------------------------------------
	RF_Balanced	0.86    	0.57    	0.69    	13.86   	0.12	        0.984
	---------------------------------------------------------------------------------------------------------
	RF_ManualWeight	0.85    	0.8	        0.82    	14.85   	0.04	        0.985
	---------------------------------------------------------------------------------------------------------
	LR_Balanced	0.94    	0.07    	0.126   	5.94    	2.31	        0.986
	---------------------------------------------------------------------------------------------------------