import switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

from datetime import datetime

class CollectTrainingStatsApp(switch.SimpleSwitch13):
    """
    SDN controller application that collects DDoS traffic statistics 
    for training the machine learning model.
    This controller collects flow statistics and labels them as DDoS (1).
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the collector application with monitoring thread.
        """
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}  # Dictionary to store active switches
        self.monitor_thread = hub.spawn(self.monitor)  # Start monitoring thread

    @set_ev_cls(ofp_event.EventOFPStateChange,[MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
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

    def monitor(self):
        """
        Periodic monitoring thread that requests flow statistics from each switch.
        Runs in an infinite loop with a 10-second interval.
        """
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)  # Wait 10 seconds between polling cycles

    def request_stats(self, datapath):
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
        Process flow statistics replies from switches and save data to CSV file.
        Labels all flows as DDoS (1).
        
        Args:
            ev: Flow statistics reply event
        """
        # Get current timestamp
        timestamp = datetime.now()
        timestamp = timestamp.timestamp()
        
        # Initialize protocol-specific variables
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        # Open file in append mode to add collected statistics
        file0 = open("FlowStatsfile.csv","a+")
        body = ev.msg.body
        
        # Process each flow with priority 1
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
                
            # Write flow stats to CSV file with label 1 (DDoS)
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond,1))
        file0.close()