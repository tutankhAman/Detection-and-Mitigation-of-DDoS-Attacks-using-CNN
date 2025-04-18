from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MachineLearning():
    """
    A class for analyzing network flow data using Logistic Regression and
    visualizing dataset characteristics to detect DDoS attacks.
    """

    def __init__(self):
        """
        Initialize the MachineLearning class and load the dataset.
        """
        print("Loading dataset ...")
        
        # Load network flow statistics dataset
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')

        # Clean the dataset by removing dots from IP addresses and flow IDs
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')        

    def flow_training(self):
        """
        Train a Logistic Regression model on flow data and visualize dataset statistics.
        Displays confusion matrix and accuracy metrics for the model as well as
        distribution of traffic types in the dataset.
        """
        print("Flow Training ...")
        
        # Prepare features (X) and target (y)
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')
        y_flow = self.flow_dataset.iloc[:, -1].values

        # Split data into training and testing sets
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Train logistic regression classifier
        classifier = LogisticRegression(solver='liblinear', random_state=0)
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        # Make predictions on test set
        y_flow_pred = flow_model.predict(X_flow_test)

        print("------------------------------------------------------------------------------")

        # Display confusion matrix and accuracy metrics
        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        print("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail*100))
        print("------------------------------------------------------------------------------")
        
        # Count normal and DDoS flows in the dataset
        benin = 0
        ddos = 0
        for i in y_flow:
            if i == 0:
                benin += 1
            elif i == 1:
                ddos += 1
                
        print("benin = ", benin)
        print("ddos = ", ddos)
        print("------------------------------------------------------------------------------")
        
        # Visualize normal vs DDoS traffic distribution
        plt.figure(figsize=(6, 6))
        plt.title("Dataset: Normal vs DDoS Traffic Distribution")
        plt.tight_layout()
        
        explode = [0, 0.1]
        plt.pie([benin, ddos], labels=['NORMAL', 'DDoS'], 
                wedgeprops={'edgecolor': 'black'},
                explode=explode, autopct="%1.2f%%")
        plt.show()
        
        # Count protocols in the dataset (ICMP, TCP, UDP)
        icmp = 0
        tcp = 0
        udp = 0
        
        proto = self.flow_dataset.iloc[:, 7].values
        proto = proto.astype('int')
        for i in proto:
            if i == 6:
                tcp += 1
            elif i == 17:
                udp += 1
            elif i == 1:
                icmp += 1

        print("tcp = ", tcp)
        print("udp = ", udp)
        print("icmp = ", icmp)
        
        # Visualize protocol distribution
        plt.figure(figsize=(6, 6))
        plt.title("Dataset: Protocol Distribution")
        
        explode = [0, 0.1, 0.1]
        plt.pie([icmp, tcp, udp], labels=['ICMP', 'TCP', 'UDP'], 
                wedgeprops={'edgecolor': 'black'},
                explode=explode, autopct="%1.2f%%")
        plt.show()
        
        # Count protocols by attack category
        icmp_normal = 0
        tcp_normal = 0
        udp_normal = 0
        icmp_ddos = 0
        tcp_ddos = 0
        udp_ddos = 0
        
        # Extract protocol and label columns
        proto = self.flow_dataset.iloc[:, [7, -1]].values
        proto = proto.astype('int')
       
        # Count each protocol by normal/DDoS category
        for i in proto:
            if i[0] == 6 and i[1] == 0:
                tcp_normal += 1
            elif i[0] == 6 and i[1] == 1:
                tcp_ddos += 1
            
            if i[0] == 17 and i[1] == 0:
                udp_normal += 1
            elif i[0] == 17 and i[1] == 1:
                udp_ddos += 1
            
            if i[0] == 1 and i[1] == 0:
                icmp_normal += 1
            elif i[0] == 1 and i[1] == 1:
                icmp_ddos += 1

        # Print protocol counts by category
        print("tcp_normal = ", tcp_normal)
        print("tcp_ddos = ", tcp_ddos)
        print("udp_normal = ", udp_normal)
        print("udp_ddos = ", udp_ddos)
        print("icmp_normal = ", icmp_normal)
        print("icmp_ddos = ", icmp_ddos)
        
        # Visualize protocol distribution by attack category
        plt.figure(figsize=(8, 8))
        plt.title("Dataset: Protocol Distribution by Traffic Type")
        
        explode = [0, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        plt.pie([icmp_normal, icmp_ddos, tcp_normal, tcp_ddos, udp_normal, udp_ddos], 
                labels=['ICMP_Normal', 'ICMP_DDoS', 'TCP_Normal', 'TCP_DDoS', 'UDP_Normal', 'UDP_DDoS'], 
                wedgeprops={'edgecolor': 'black'}, explode=explode, autopct="%1.2f%%")
        plt.show()
    
def main():
    """
    Main function to execute the machine learning pipeline.
    Tracks execution time for performance measurement.
    """
    start = datetime.now()
    
    ml = MachineLearning()
    ml.flow_training()

    end = datetime.now()
    print("Training time: ", (end-start)) 

if __name__ == "__main__":
    main()