from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MachineLearning():
    """
    A class for comparing multiple machine learning algorithms on network flow data
    for DDoS attack detection.
    """

    def __init__(self):
        """
        Initialize the MachineLearning class, load dataset and prepare data for training.
        """
        print("Loading dataset ...")
        
        # Counter to track which algorithm is being evaluated
        self.counter = 0
        
        # Load flow statistics dataset
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')

        # Clean the dataset by removing dots from IP addresses and flow IDs
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')
        
        # Extract features (X) and target labels (y)
        self.X_flow = self.flow_dataset.iloc[:, :-1].values
        self.X_flow = self.X_flow.astype('float64')
        self.y_flow = self.flow_dataset.iloc[:, -1].values

        # Split data into training and testing sets - done once for all classifiers
        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(
            self.X_flow, self.y_flow, test_size=0.25, random_state=0)

    def LR(self):
        """
        Train and evaluate Logistic Regression classifier.
        """
        print("------------------------------------------------------------------------------")
        print("Logistic Regression ...")

        self.classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.Confusion_matrix()
        
    def KNN(self):
        """
        Train and evaluate K-Nearest Neighbors classifier.
        """
        print("------------------------------------------------------------------------------")
        print("K-NEAREST NEIGHBORS ...")

        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.Confusion_matrix()
 
    def SVM(self):
        """
        Train and evaluate Support Vector Machine classifier.
        """
        print("------------------------------------------------------------------------------")
        print("SUPPORT-VECTOR MACHINE ...")

        self.classifier = SVC(kernel='rbf', random_state=0)
        self.Confusion_matrix()
        
    def NB(self):
        """
        Train and evaluate Naive Bayes classifier.
        """
        print("------------------------------------------------------------------------------")
        print("NAIVE-BAYES ...")

        self.classifier = GaussianNB()
        self.Confusion_matrix()
        
    def DT(self):
        """
        Train and evaluate Decision Tree classifier.
        """
        print("------------------------------------------------------------------------------")
        print("DECISION TREE ...")

        self.classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.Confusion_matrix()
        
    def RF(self):
        """
        Train and evaluate Random Forest classifier.
        """
        print("------------------------------------------------------------------------------")
        print("RANDOM FOREST ...")

        self.classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.Confusion_matrix()
        
    def Confusion_matrix(self):
        """
        Calculate confusion matrix for the current classifier and update the comparative visualization.
        Builds a bar chart comparing all classifiers as they are evaluated.
        """
        self.counter += 1
        
        # Train model with current classifier
        self.flow_model = self.classifier.fit(self.X_flow_train, self.y_flow_train)

        # Make predictions on test set
        self.y_flow_pred = self.flow_model.predict(self.X_flow_test)

        print("------------------------------------------------------------------------------")

        # Generate and display confusion matrix
        print("confusion matrix")
        cm = confusion_matrix(self.y_flow_test, self.y_flow_pred)
        print(cm)

        # Calculate and display accuracy metrics
        acc = accuracy_score(self.y_flow_test, self.y_flow_pred)
        print("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail*100))
        print("------------------------------------------------------------------------------")
        
        # Set up for side-by-side bar chart of all classifiers
        x = ['TP','FP','FN','TN']
        x_indexes = np.arange(len(x))
        width = 0.10
        plt.xticks(ticks=x_indexes, labels=x)
        plt.title("Résultats des algorithmes")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")
        
        # Add results for current classifier to the chart
        # Each classifier gets a different position and color
        if self.counter == 1:  # Logistic Regression
            y1 = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
            plt.bar(x_indexes-2*width, y1, width=width, color="#1b7021", label='LR')
            plt.legend()
        if self.counter == 2:  # KNN
            y2 = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
            plt.bar(x_indexes-width, y2, width=width, color="#e46e6e", label='KNN')
            plt.legend()
        if self.counter == 3:  # Naive Bayes
            y3 = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
            plt.bar(x_indexes, y3, width=width, color="#0000ff", label='NB')
            plt.legend()
        if self.counter == 4:  # Decision Tree
            y4 = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
            plt.bar(x_indexes+width, y4, width=width, color="#e0d692", label='DT')
            plt.legend()
        if self.counter == 5:  # Random Forest
            y5 = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
            plt.bar(x_indexes+2*width, y5, width=width, color="#000000", label='RF')
            plt.legend()
            plt.show()  # Show final comparison chart after all classifiers are evaluated
        
def main():
    """
    Main function to execute the ML comparison pipeline.
    Tracks execution time for each algorithm and overall script.
    """
    start_script = datetime.now()
    
    ml = MachineLearning()
    
    # Train and evaluate Logistic Regression
    start = datetime.now()
    ml.LR()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end-start)) 
    
    # Train and evaluate KNN
    start = datetime.now()
    ml.KNN()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end-start))
    
    # SVM is commented out, likely due to longer training time
    # start = datetime.now()
    # ml.SVM()
    # end = datetime.now()
    # print("LEARNING and PREDICTING Time: ", (end-start))
    
    # Train and evaluate Naive Bayes
    start = datetime.now()
    ml.NB()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end-start))
    
    # Train and evaluate Decision Tree
    start = datetime.now()
    ml.DT()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end-start))
    
    # Train and evaluate Random Forest
    start = datetime.now()
    ml.RF()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end-start))
    
    end_script = datetime.now()
    print("Script Time: ", (end_script-start_script))

if __name__ == "__main__":
    main()