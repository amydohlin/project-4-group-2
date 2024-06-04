# project-4-group-2
Final bootcamp project analyzing liver cirrhosis data and using machine learning to predict cirrhosis stages.  
Members: Amy Dohlin, Anna Bitzer, Tianyue Li, Christine Jauregui

Topic: Liver Cirrhosis Stage Prediction  
Dataset: Liver Cirrhosis Stage Classification 🩺 (kaggle.com) (25,000+ rows)  
Goal: Create, train and deploy a neural network machine learning model that can predict liver cirrhosis stage based on patient lab result data.   
Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.  

__PHASE I: Data Cleaning__  
- With a JupyterNotebook in VSCode, we read in the liver_cirhosis.cvs file into a dataframe. 
- Next we checked for null values in the liver_cirhosis.csv file and removed any rows with null values.  
- We also checked the data types for the columns in the dataframe and used the drop.duplicates function to remove any repeated rows.  
-  We wrote the data in this modified dataframe to a csv file, liver_clean.csv.   

__PHASE 2: Preliminary Visualizations__   
- Using a Jupyter Notebook and MatplotLib, we created overviews and summary statistics of the data in liver_clean.csv.
- We also identified outliers in the data in order to drop outlying rows from the dataframe before modeling the data.

__PHASE 3: Data Transformation__    
- We read the liver_clean csv into a Google Colab notebook, where we saved the data into a Pandas dataframe.
- We generated a list of six categorial variables (.dtypes == "object") in order to run a single instance of OneHoteEncoder.
- We created a dataframe of the encoded columns and their values, which all became 0.0 or 1.0. The number of features in the dataframe was now 27 at this point. 
- We defined y as the values in the 'Stage' columns, whose values are all 1, 2, or 3.
- We defined X as the values in all other columns of the dataframe (except 'Stage').
- To prepare for the neural networks models, we split the preprocessd data into training and testing datasets for X and y.
- Next, we used the StandardScalar method to scale the X_train and X_test data.

__PHASE 4: Neural Network Models__   
Neural Network Model #1  
- For the first iteration of the neural network model, we set the number of input features to be equal to the shape of the scaled X_train dataset.  
- After defining a keras.Sequential() model, we added a first hidden layer with 8 nodes and a second hidden layer with 5 nodes. For the output layer we set the activation to 'sigmoid.' Hidden Layers 1 and 2 were activated with "relu".
- After fitting, compiling, and training the model (100 epochs), we noted an accuracy score of 32.6%.  

Neural Network Model #2  
- For the second iteration of the neural network model, we doubled the nodes in the hidden layers to 16 and 10, respectively. All other layer characteristics remained the same as before.
- This version of the model returned an accuracy score of 46.3% after 100 epochs.  
  
Neural Network Model #3  
- The third iteration of the model was similar to the previous iteration, but two additional hidden layers with 12 nodes each were included.
- The two new hidden layers also had activations of "relu".
- This model version produced an accuracy score of 46.6%.

As none of these neural network models achieved high accuracy scores, we decided to try Principal Component Analysis to reduce the number of features. 

*---- insert description of Tianyue's optimizations and the results on the model. -------*

__PHASE 5: Random Forest Models__   
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

Techniques/Skills to Use:   
Pandas  
Tableau  
Matplotlib  
SQL / Spark  
Softmax  
Data Bricks?  


