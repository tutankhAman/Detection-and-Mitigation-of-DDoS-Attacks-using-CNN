from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):
    """
    SDN controller application that uses Random Forest to detect DDoS attacks.
    Extends basic switch functionality with flow statistics monitoring and 
    real-time DDoS attack detection.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize controller, start monitoring thread, and train the ML model.
        """
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}  # Dictionary to store active switches
        self.monitor_thread = hub.spawn(self._monitor)  # Start monitoring thread

        # Train the Random Forest model and measure training time
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """
        Track the addition and removal of switches.
        
        Args:
            ev: OpenFlow event containing datapath information
        """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        """
        Periodic monitoring thread that requests statistics and predicts traffic type.
        Runs in an infinite loop with a 10-second interval.
        """
        while True:
            # Request stats from each registered switch
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)  # Wait 10 seconds between polling cycles

            # Run prediction on collected flow statistics
            self.flow_predict()

    def _request_stats(self, datapath):
        """
        Send flow statistics request to a switch.
        
        Args:
            datapath: Switch to request statistics from
        """
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """
        Process flow statistics replies from switches and save data for prediction.
        
        Args:
            ev: Flow statistics reply event
        """
        # Get current timestamp
        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        # Create/overwrite the prediction data file with headers
        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        
        # Initialize protocol-specific variables
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        # Process each flow entry with priority 1
        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
        
            # Extract IP addresses and protocol
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            # Handle different IP protocols (ICMP, TCP, UDP)
            if stat.match['ip_proto'] == 1:  # ICMP
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
            elif stat.match['ip_proto'] == 6:  # TCP
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']
            elif stat.match['ip_proto'] == 17:  # UDP
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            # Create flow identifier from 5-tuple
            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
          
            # Calculate per-second statistics, handling division by zero
            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
                packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
                
            try:
                byte_count_per_second = stat.byte_count/stat.duration_sec
                byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
                
            # Write flow data to the prediction file (note: no label column)
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
            
        file0.close()

    def flow_training(self):
        """
        Train the Random Forest machine learning model using collected flow statistics.
        Evaluates model performance using confusion matrix and accuracy metrics.
        """
        self.logger.info("Flow Training ...")

        # Load training dataset (contains both benign and DDoS traffic)
        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        # Clean the dataset by removing dots from IP addresses and flow IDs
        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        # Prepare features (X) and target (y) 
        X_flow = flow_dataset.iloc[:, :-1].values  # All columns except last
        X_flow = X_flow.astype('float64')
        y_flow = flow_dataset.iloc[:, -1].values  # Last column (label)

        # Split data into training and testing sets
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(
            X_flow, y_flow, test_size=0.25, random_state=0)

        # Create and train Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        # Predict on test data to evaluate model
        y_flow_pred = self.flow_model.predict(X_flow_test)

        # Display evaluation metrics
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        """
        Use the trained model to predict whether current network traffic contains DDoS attacks.
        Identifies potential attack victims when DDoS traffic is detected.
        """
        try:
            # Load the prediction data file
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            # Clean the dataset by removing dots from IP addresses and flow IDs
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            # Prepare data for prediction
            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            # Make predictions using the trained model
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            # Count legitimate and DDoS traffic predictions
            legitimate_trafic = 0
            ddos_trafic = 0

            # Process each prediction
            for i in y_flow_pred:
                if i == 0:  # Benign traffic
                    legitimate_trafic = legitimate_trafic + 1
                else:  # DDoS traffic
                    ddos_trafic = ddos_trafic + 1
                    # Identify potential victim host based on destination IP
                    # Modulo 20 to map to host ID in the range 0-19
                    victim = int(predict_flow_dataset.iloc[i, 5]) % 20

            # Log results and alert if DDoS is detected
            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic/len(y_flow_pred)*100) > 80:  # If more than 80% traffic is legitimate
                self.logger.info("legitimate trafic ...")
            else:  # DDoS attack detected
                self.logger.info("ddos trafic ...")
                self.logger.info("victim is host: h{}".format(victim))
            self.logger.info("------------------------------------------------------------------------------")
            
            # Clean up prediction file (prepare for next round)
            file0 = open("PredictFlowStatsfile.csv","w")
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except:
            # Handle errors silently (e.g., empty prediction file)
            pass