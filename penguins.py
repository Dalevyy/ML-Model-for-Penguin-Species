import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# kFold: Number of folds we will use
kFold = 10
# penguinLabels: Three main classes for our dataset
penguinLabels = ['Adelie', 'Gentoo', 'Chinstrap']

'''
Description: Reads in the penguins dataset and preprocesses
the data before it can be used in any of our models
Parameters: None
Return Type: Two lists (one for features, one for labels)
'''
def preProcess():
    # normColNames: Columns that will be normalized
    normColNames = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # Read in the data from the csv file
    penguinData = pd.read_csv("palmerpenguins_original.csv")
    
    # First, we will remove the samples with missing values
    # and we will drop the year of observation as a feature
    penguinData = penguinData.dropna()
    penguinData = penguinData.drop("year", axis=1)

    # Then, we will normalize the data using z-scores
    for col in normColNames:
        penguinData[col] = zscore(penguinData[col])

    # We then convert out categorical variables into nominal variables
    # using one-hot encoding
    penguinData = pd.get_dummies(data=penguinData,columns=['island', 'sex'], dtype=int)
    penguinData = penguinData.drop(["island_Torgersen", "sex_male"], axis = 1)
    
    # We will also randomize the placement of our data
    # in the dataframe
    penguinData = penguinData.sample(frac=1).reset_index(drop=True)

    # We separate our labels from the features, so we 
    # can use them in our ML algos
    penguinLabels = penguinData.iloc[:,0]
    penguinData = penguinData.iloc[:,1:]

    # Finally, we convert our labels into numerical values (0,1,2)
    # This is primarily needed for the neural network implementation
    penguinLabelsEncoder = LabelEncoder()
    penguinLabels = pd.DataFrame(penguinLabelsEncoder.fit_transform(penguinLabels))

    return penguinData, penguinLabels

'''
Description: Function that prints out the accuracy for 
the current fold
Parameters:
Return Type: None
'''
def printAcc(modelName, foldNum, accVal):
    # If we are printing out the first fold, we will print
    # the model's name
    if(foldNum == 1):
        print(f"\n{modelName}\n")
    print(f"Fold {foldNum} Testing Accuracy: {accVal:0.4f}%")

'''
Description: Function that creates a neural network
Parameters: Dataframes containing our features and labels
respectively with the folds we are using
Return Type: List that has the average accuracies per fold
'''
def neuralNetwork(features, labels, folds):
    accuracies = []
    currFold = 1
    aggregate_confusion_matrix = 0
    for train, test in folds.split(features):
        # First, we will split our data into training and testing
        # sets respectively 
        featuresTrain= features.iloc[train, :]
        labelsTrain = labels.iloc[train]

        featuresTest = features.iloc[test, :]
        labelsTest = labels.iloc[test]

        # Then, we will set up our neural network
        # We will use Sequential to create 4 layers: 1 input layer
        # 2 hidden layers, and then a final output layer
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(3)])
        
        # Next, we define our loss and optimizer functions before 
        # creating our neural network
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
              optimizer=opt,
              loss=loss_fn,
              metrics=["accuracy"])
        
        # Finally, we fit our neural network to the training data and then
        # evaluate it using the testing datahy
        trainNN = model.fit(featuresTrain, labelsTrain, batch_size=2, epochs=10, verbose=0)
        
        # Making Predictions for Confusion Matrix
        labelsPredicted = np.argmax(model.predict(featuresTest, batch_size=2, verbose=0), axis = 1)
        accNN = accuracy_score(labelsTest, labelsPredicted)
        accuracies.append(accNN)
        aggregate_confusion_matrix += confusion_matrix(labelsTest, labelsPredicted)

        printAcc("Neural Network", currFold, (accNN*100))
        currFold += 1

    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguinLabels, columns=penguinLabels)
    return accuracies, average_confusion_matrix

'''
Description: Function that creates a logistic regression model
Parameters: Features and labels dataframes with the folds we are using
Return Type: List that contains our average accuracies per fold
'''
def logisticRegression(features, labels, folds):
    accuracies = []
    # First, create our logistic regression model
    lrClassifier = LogisticRegression(multi_class = "multinomial", solver = "saga", max_iter = 100)
    aggregate_confusion_matrix = 0

    for i, (training_ind, testing_ind) in enumerate(folds.split(features)):
        # We will get our training dataset and fit it into the logistic regression model
        penguinFeatures_split = features[training_ind[0] : training_ind[training_ind.size-1]]    
        penguinLabels_split = labels[training_ind[0] : training_ind[training_ind.size-1]]

        lrClassifier.fit(penguinFeatures_split, np.array(penguinLabels_split).ravel())

        # We now get our testing dataset and use the testing features to predict the labels
        # for our samples
        penguinFeatures_test_split = features[testing_ind[0] : testing_ind[testing_ind.size - 1]]
        penguinLabels_test_split = labels[testing_ind[0] : testing_ind[testing_ind.size - 1]]

        penguinLabels_pred = lrClassifier.predict(penguinFeatures_test_split)
        aggregate_confusion_matrix += confusion_matrix(penguinLabels_test_split, penguinLabels_pred)
        
        # We then compute the accuracy for our current fold
        acc = accuracy_score(penguinLabels_test_split, penguinLabels_pred)
        accuracies.append(acc)
        printAcc("Logistic Regression", i+1, (acc*100))
        
    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguinLabels, columns=penguinLabels)
    return accuracies, average_confusion_matrix

'''
Description: Function that creates a SVM 
Parameters: Features and labels dataframes with folds
Return Type: Average accuracies list
'''
def SVM(features, labels, folds):
    accuracies = []
    currFold = 1
    aggregate_confusion_matrix = 0

    for train, test in folds.split(features):
        # Split the data into training and testing sets
        featuresTrain= features.iloc[train, :]
        labelsTrain = labels.iloc[train]

        featuresTest = features.iloc[test, :]
        labelsTest = labels.iloc[test]
        
        # Set up our SVM by using a kernel linear function
        # and fit our data into it
        svmModel = SVC(kernel='linear')
        svmModel.fit(featuresTrain, np.array(labelsTrain).ravel())

        # Predict what the labels will be for our testing data
        # and compute the accuracy 
        labelsPredict = svmModel.predict(featuresTest)
        aggregate_confusion_matrix += confusion_matrix(labelsTest, labelsPredict)
        
        testAccuracy = accuracy_score(labelsTest, labelsPredict) * 100
        accuracies.append(testAccuracy)
        printAcc("Support Vector Machine", currFold, testAccuracy)
        currFold += 1
    
    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguinLabels, columns=penguinLabels)
    return accuracies, average_confusion_matrix

'''
Description: Function that creates a kNN model
Parameters: Dataframes for our features and labels 
alongside the folds we are using
Return Type: List containing the average
accuracies for each fold
'''
def kNN(features, labels, folds):
    accuracies = []
    currFold = 1
    aggregate_confusion_matrix = 0

    for train, test in folds.split(features):
        # First, split up the data into training/testing sets
        featuresTrain= features.iloc[train, :]
        labelsTrain = labels.iloc[train]

        featuresTest = features.iloc[test, :]
        labelsTest = labels.iloc[test]

        # Then, we will create our kNN model and fit the training 
        # data into our model
        knnModel = KNeighborsClassifier(n_neighbors=3)
        knnModel.fit(featuresTrain, np.array(labelsTrain).ravel())
        
        # Finally, we predict what labels the testing set will have
        # and we will then compute the accuracy
        labelsPredict = knnModel.predict(featuresTest)
        aggregate_confusion_matrix += confusion_matrix(labelsTest, labelsPredict)
        
        testAccuracy = accuracy_score(labelsTest, labelsPredict) * 100
        accuracies.append(testAccuracy)
        printAcc("k-Nearest Neighbor", currFold, testAccuracy)
        currFold += 1

    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguinLabels, columns=penguinLabels)
    return accuracies, average_confusion_matrix

'''
Description: Main function for our ML project
Parameters: None
Return Type: None
'''
def main():
    # First, call preProcess so our data can be preprocessed
    penguinFeatures, penguinLabels = preProcess()
    
    # Then, we will set up our training/testing sets
    folds = KFold(n_splits=kFold)

    print("Palmer Penguins Dataset - Model Accuracies")
    # We will call separate ML functions to do our 4 total algos
    accuraciesNN, average_cm_NN = neuralNetwork(penguinFeatures, penguinLabels, folds)
    accuraciesLR, average_cm_LR = logisticRegression(penguinFeatures, penguinLabels, folds)
    accuraciesSVM, average_cm_SVM = SVM(penguinFeatures, penguinLabels, folds)
    accuraciesKNN, average_cm_KNN = kNN(penguinFeatures, penguinLabels, folds)
    
    print("\nMean Accuracies\n")
    # We then display the average accuracy for each ML implementation we wrote
    print(f"Mean Accuracy for Neural Network Implementation: {np.mean(np.array(accuraciesNN))*100:.4f}%")
    print(f"Mean Accuracy for Logistic Regression Implementation: {np.mean(np.array(accuraciesLR))*100:.4f}%")
    print(f"Mean Accuracy for SVM Implementation: {np.mean(np.array(accuraciesSVM)):.4f}%")
    print(f"Mean Accuracy for kNN Implementation: {np.mean(np.array(accuraciesKNN)):.4f}%")

    # Finally, we will set up our confusion matrices, so we can print it out
    # to the user 
    figure, axes = plt.subplots(1,4, figsize=(24,5))
    confusion_matrices = [average_cm_NN, average_cm_LR, average_cm_SVM, average_cm_KNN]
    confusion_matrices_titles = ['NeuralNetwork', "Log.Regression", "SVM", "KNN"]
    accuracies = [np.mean(np.array(accuraciesNN))*100,np.mean(np.array(accuraciesLR))*100,
                  np.mean(np.array(accuraciesSVM)),np.mean(np.array(accuraciesKNN))]
    
    # We will print each confusion matrix side by side for each model
    for axis, cm, title in zip(axes, confusion_matrices, confusion_matrices_titles):
        sns.heatmap(cm, annot=True, fmt='f', cmap='Blues', ax=axis, cbar=False)
        axis.set_title(title)
        axis.set_xlabel('Predicted', fontsize=12)
        axis.set_ylabel('Actual', fontsize=12)
        axis.set_aspect('equal')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

if __name__ == "__main__":
    main()
