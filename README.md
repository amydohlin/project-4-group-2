# project-4-group-2
Final bootcamp project analyzing liver cirrhosis data and using machine learning to predict cirrhosis stages.  
Members: Amy Dohlin, Anna Bitzer, Tianyue Li, Christine Jauregui

### Topic: Liver Cirrhosis Stage Prediction  
### Dataset: Liver Cirrhosis Stage Classification 🩺 (kaggle.com) (25,000+ rows)  
Goal: Create, train and deploy a neural network machine learning model that can predict liver cirrhosis stage based on patient lab result data.   

### What is Liver Cirrhosis?  
Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.  

### Attribute Information:    
N_Days: Number of days between registration and the earlier of death, transplantation, or study analysis time in 1986  
Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)  
Drug: type of drug D-penicillamine or placebo  
Age: age in days  
Sex: M (male) or F (female)  
Ascites: presence of ascites N (No) or Y (Yes)  
Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)  
Spiders: presence of spiders N (No) or Y (Yes)  
Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)  
Bilirubin: serum bilirubin in [mg/dl]  
Cholesterol: serum cholesterol in [mg/dl]  
Albumin: albumin in [gm/dl]  
Copper: urine copper in [ug/day]  
Alk_Phos: alkaline phosphatase in [U/liter]  
SGOT: SGOT in [U/ml]  
Tryglicerides: triglicerides in [mg/dl]  
Platelets: platelets per cubic [ml/1000]  
Prothrombin: prothrombin time in seconds [s]  
Stage: histologic stage of disease ( 1, 2, or 3 )  

### __PHASE I: Data Cleaning__  
- With a JupyterNotebook in VSCode, we read in the liver_cirhosis.cvs file into a dataframe. 
- Next we checked for null values in the liver_cirhosis.csv file and removed any rows with null values.  
- We also checked the data types for the columns in the dataframe and used the drop.duplicates function to remove any repeated rows.  
![07](https://github.com/amydohlin/project-4-group-2/assets/151464511/a7a39afd-cd03-4cb9-b4c1-dc1bf76f783e)  
- After removing duplicate records, the file dropped from 25,000 to 10,000 rows.   
-  We wrote the data in this modified dataframe to a csv file, liver_clean.csv.   

### __PHASE 2: Preliminary Visualizations__   
- Using a Jupyter Notebook and MatplotLib, we created overviews and summary statistics of the data in liver_clean.csv.  

- We also identified outliers in the data in order to drop outlying rows from the dataframe before modeling the data.
- We explored the relationship between 'Stage' and 'Number of Months' between registration and the earlier of death, transplantation, or study analysis time in 1986. The latter feature we calculated from the N_Days feature (Number of days between registration and the earlier of death, transplantation, or study analysis time in 1986).
- The results were unsurprising. The lower the stage of Cirrhosis, the more time elapsed between patient registration and death, transplantation, or study analysis time. The more advanced the stage of the disease, the sooner the onset of death or liver transplantation.   
![08](https://github.com/amydohlin/project-4-group-2/assets/151464511/1cab0863-ee66-44b7-bedb-c2d73a561498)
![09](https://github.com/amydohlin/project-4-group-2/assets/151464511/f5d97fa4-ef32-4f8c-984a-8bb74325df25)

### __PHASE 3: Data Transformation__    
- We read the liver_clean csv into a Google Colab notebook, where we saved the data into a Pandas dataframe.
- We generated a list of six categorial variables (.dtypes == "object") in order to run a single instance of OneHoteEncoder.
- We created a dataframe of the encoded columns and their values, which all became 0.0 or 1.0. The number of features in the dataframe was now 27 at this point. 
- We defined y as the values in the 'Stage' columns, whose values are all 1, 2, or 3.
- We defined X as the values in all other columns of the dataframe (except 'Stage').
- To prepare for the neural networks models, we split the preprocessd data into training and testing datasets for X and y.
- Next, we used the StandardScalar method to scale the X_train and X_test data.

### __PHASE 4: Neural Network Models__   
**Neural Network Model #1**
- For the first iteration of the neural network model, we set the number of input features to be equal to the shape of the scaled X_train dataset.  
- After defining a keras.Sequential() model, we added a first hidden layer with 8 nodes and a second hidden layer with 5 nodes. For the output layer we set the activation to 'sigmoid.' Hidden Layers 1 and 2 were activated with "relu".
- After fitting, compiling, and training the model (100 epochs), we noted an accuracy score of 32.6%.  

**Neural Network Model #2**
- For the second iteration of the neural network model, we doubled the nodes in the hidden layers to 16 and 10, respectively. All other layer characteristics remained the same as before.
- This version of the model returned an accuracy score of 46.3% after 100 epochs.  
  
**Neural Network Model #3**  
- The third iteration of the model was similar to the previous iteration, but two additional hidden layers with 12 nodes each were included.
- The two new hidden layers also had activations of "relu".
- This model version produced an accuracy score of 46.6%.

###__PHASE 5: Model Optimization__  
- As none of the neural networks achieved high accuracy scores, we used KerasTuner to decide    
a) which activation function to use  
b) the number of neurons in the first layer  
c) the number of hidden layers and neurons in the layers  

![image](https://github.com/amydohlin/project-4-group-2/assets/42381263/93cebdc3-1cf6-4ebd-952f-f128f138cc64)



*-----Include things about results of KerasTuner ------*
- In our evaluation of the optimized model, we created a confusion matrix and generated a correlation heatmap. 
- The confusion matrix ...
- 
- The correlation heatmap indicated the strongest positive correlation (0.65) between Spiders_N and Ascites_N and between Spiders_Y and Ascites_Y. This is to be expected as Spider angiomas tend to appear in patients with chronic liver disease and ascites.
- The next highest positive correlation (0.42) is observed between Bilirubin and Status_D (Death). This is unsurprising as high bilirubin levels in the blood are indicative of improperly functioning liver, a risk factor for death in people with cirrhosis.
- We also noticed a positive correlation of 0.39 between Ascites_N and Drug_Penicillamine and a positive correlation of 0.39 between Ascites_Y and Drug_Placebo. This appears logical because Penicillamine is a drug used to treat ascites. It stands to reason that patients receiving the Placebo instead of Penicillamine would be more likely to test positive for Ascites while those recieving the Penicillamine would be more likely to test negative for Ascites. 
![06](https://github.com/amydohlin/project-4-group-2/assets/151464511/0412eaaf-0230-4a15-8f6d-fb51b14700f9)

###__PHASE 6: Random Forest Models__   
We decided to try using a Random Forest Model on the data because they are robust against overfitting, robust to outliers and non-linear data, and efficient on large databases.  

- As the process of reading the cvs file into the dataframe created a new index identical to the the Unnamed column, we dropped the latter and the 'Stage' feature from our definition of X.
- The y variable remained as 'Stage'. 
- Next we identified the categorical features in the dataframe by getting the counts of each value.
- Because Edema had three values, not just Y or N like the other categorical fields, we set Edema aside to later apply the "get dummies" function.
- For the categorical features with only two values, we replaced the Y values with 0s, the N values with 1s, the D-penicillamine value in Drug with 1s, and the Placebo value in Drug with 0. We set all new values to integer type.
- We used the "get dummies" function to encode the remaining categorical variables, Status and Edema, and turn their values into integers.
- After splitting the data into training and testing sets, we scaled the X_train, X_test, y_train, and y_test sets.
- Then, we created a random forest classifier with 500 estimators and 42 random states.
- After fitting the model on X_train_scaled and y_train, we used the model to make predictions with the test data, X_test_scaled.
- We calculated and displayed the model's accuracy score and classification report, and used the seaborn heatmmap to produce a confusion matrix. 

![04](https://github.com/amydohlin/project-4-group-2/assets/151464511/1aa9b443-5e6e-4ecc-b153-a5f1a0a10197)

- At 86.8%, the overall accuracy score was a significant improvement over the neural network models' accuracies. 
- Precision and recall were also high, ranging from 83% to 91% and 86% to 91%, respectively
- The Random Forest Model did a good job at classifying Stage 1, Stage 2, and Stage 3 liver cirrhosis.
- Next, we evaluated the importance of the various features in the data. The categorical features, such as 'Spiders', 'Ascites', 'Edema', 'Sex', and 'Status', had the least importance to the model's classification power. Among the non-categorical features, the features we scaled, one group stood out as most imporant. That group included Prothrombin, Platelets, Albumin, N-Days, Age, and Bilirubin. A second group of non-categorical features, including Copper, SGOT, Alk_Phos, Cholesterol, Hepatomegaly, and Tryglicirides, was identified as moderately important. 

![05](https://github.com/amydohlin/project-4-group-2/assets/151464511/cd661e6b-d6d3-45d0-9b83-9d8004675f46)

With the classification results of the Random Forest Model we plotted a series of bar charts  in Tableau to visualize the importance of various features on the determindation of 'Stage'. 

__Prothrombin, feature importance 0.13__  
According to the Mayo Clinic, the protein Prothrombin is one of many factors in the blood that help it to clot. Prothrombin time is the number of seconds it takes for a patient's blood to clot. The Prothrombin time test is one of many tests used to check people for liver disease and to screen people waiting for liver transplants. Blood normally clots in 10 to 13 seconds. Liver damage is one cause of blood clotting slower than average.   

- In the count of observations versus Prothrombin time graph, we see that most patients -- including those with Stage 1, Stage 2 and Stage 3 of the disease -- are within the normal range of 10 to 13 seconds.  
- Within that range, Stage 3 patients increase in counts as Prothrombin time increases.   
- Meanwhile, Stage 1 and Stage 2 patients appear more heavily concentrated at the lower end of the normal range.   
- These obervations align with the fact that Prothrombin time is one, but not the only, factor in determining blood clotting speed.  
- They also align with the fact that the blood's ability to clot decreases as disease worsens.   

__Platelets, feature importance 0.11__  
Johns Hopkins School of Medicine describes a normal platelet count as 150,000 to 450,000 platelets per microliter of blood. People with less than 150,000 have a condition called thrombocytopenia. This condition makes it more difficult for blood to clot. Medical researchers have documented thrombocytopenia in up to 76% of patients with chronic liver disease, and approximately 13% of patients with cirrhosis. The condition can postpone or interefere with diagnostic and therapeutic procedures, as it increases the risk of bleeding in patients.   
  
- In the Platelets graph we can see that most of the Stage 1, Stage 2, and Stage 3 patients have platelet counts within the normal range.  
- At the same time, however, a large segment of Stage 3 patients are clustered at or around the 150,000 point.  
- As in the Prothrombin graph, we don't see dramatically abnormal test resuls but we do see noticeable differences between patients stages.  

__Albumin, feature importance 0.11__  
According to the National Kidney Foundation, 3.5 g/dL to 5 g/dL (grams per deciliter) is the normal level of the protein albumin found in blood plasma. The University of Rochester Medical Center explains that the liver makes albumin and that lower levels of this protein in the blood can indicate liver disease. Per the University of California San Francisco Health, lower-than-normal albumin can accompany the presence of ascites, or fluid build up in the abdomen. 

- In the Albumin chart, Stage 1 and Stage 2 patients appear mostly within the normal range.
- Stage 3 patients are more clustered around the lower end of the normal range and also exhibit a longer left-hand-side "tail" in their distribution. This means Stage 3 patients extend far beyond the lower limits of the normal range.
- These observations are in line with the medical evidence that albumin levels decrease as liver disease advances.  

__Number of Days to Study End, feature importance 0.10__   
- The study end was marked by whichever came first of death, transplantation, or study analysis time.
- Stage 3 patients trended towards lower number of days in the study.
- Stage 2 patients clustered towards the middle of number of days in the study.
- Stage 1 patients were relatively evenly distributed across the range of days.
- These results suggest that the later the stage of cirrhosis, the more likely that patients would end the study early - perhaps due to death or transplantation.

__Age, feature importance 0.09__   
- Study patients ranged in age from 25 to 80 years old.  
- The Stage 1 patients were predominantly in their early 40s to early 60s.
- The Stage 2 patients were largely in their early 40s to early 60s.
- The bulk of the Stage 3 patients were between mid 40s and early 60s.
- Among patients aged mid 60s to early 70s, about half were Stage 3.
- This suggests that chronic liver disease can begin at Stage 1 at relatively young ages; it's not a disease one acquires simply by aging.

![Tableau Top 5 Random Forest Features](https://github.com/amydohlin/project-4-group-2/assets/151464511/0c4e03ce-8b5f-48c9-b712-fbfadeb8059e)



Create Training Model (Google Colab, incorporate SQL/Spark): Amy, Christine  
Look at models like random forest, neural networks, etm.  
Categorical vs discrete variable cleaning (one-hot encoding)  
Drop irrelevant columns, if any  
Training vs test data  
Choose various neural network parameters  
Optimize Model: Tianyue, Anna  
Use PCA to eliminate less-impactful factors to make our model concise and strong  
Analyze the model accuracy using confusion matrix and classification report  
Use tensorflow to optimize the activation function  
Document changes along the way, and effect on accuracy  
Resulting Visualizations (Tableau?): Group  
Highlight the findings of our neural network   
ReadMe: Group  
Presentation: Anna	  

### Techniques/Skills to Use:   
* Pandas  
* Tableau  
* Matplotlib  
* SQL / Spark  
* Softmax  

### References & Acknowledgements:
* We would like to thank our instructor, Hunter Hollis and teaching assistant Randy for their guidance throughout this project.
* Model optimization for multi-category output: https://stackoverflow.com/questions/61550026/valueerror-shapes-none-1-and-none-3-are-incompatible
