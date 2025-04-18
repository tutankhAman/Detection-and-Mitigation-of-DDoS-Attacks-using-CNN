from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MachineLearning():
    """
    A class for analyzing network flow data using Random Forest classifier
    and visualizing results for DDoS attack detection.
    """

    def __init__(self):
        """
        Initialize the MachineLearning class and load the flow statistics dataset.
        """
        print("Loading dataset ...")
        
        # Load flow statistics dataset
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')

        # Clean the dataset by removing dots from IP addresses and flow IDs
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')       

    def flow_training(self):
        """
        Train a Random Forest classifier on flow data and evaluate its performance.
        Creates visualization of confusion matrix components (TP, FP, FN, TN).
        """
        print("Flow Training ...")
        
        # Extract features (X) and target labels (y)
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')
        y_flow = self.flow_dataset.iloc[:, -1].values

        # Split data into training and testing sets
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(
            X_flow, y_flow, test_size=0.25, random_state=0)

        # Train Random Forest classifier with 10 trees using entropy criterion
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        # Predict on test data
        y_flow_pred = flow_model.predict(X_flow_test)

        print("------------------------------------------------------------------------------")

        # Generate and display confusion matrix
        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        # Calculate and display accuracy metrics
        acc = accuracy_score(y_flow_test, y_flow_pred)
        print("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail*100))
        print("------------------------------------------------------------------------------")
        
        # Visualize confusion matrix components
        x = ['TP','FP','FN','TN']
        plt.title("Random Forest")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")
        y = [cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
        plt.bar(x, y, color="#000000", label='RF')
        plt.legend()
        plt.show()
    
def main():
    """
    Main function to execute the Random Forest training pipeline.
    Tracks and displays execution time.
    """
    start = datetime.now()
    
    ml = MachineLearning()
    ml.flow_training()

    end = datetime.now()
    print("Training time: ", (end-start)) 

if __name__ == "__main__":
    main()