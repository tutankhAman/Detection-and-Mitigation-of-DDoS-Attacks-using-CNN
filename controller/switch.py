from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp

class SimpleSwitch13(app_manager.RyuApp):
    """
    Basic OpenFlow v1.3 switch implementation for the SDN controller.
    Handles packet forwarding, MAC learning, and flow rule installation.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        """
        Initialize the switch with an empty MAC-to-port mapping table.
        """
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}  # Dictionary to store MAC address to port mappings

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handler called when a switch connects to the controller.
        Installs a default flow entry to send unmatched packets to the controller.
        
        Args:
            ev: Switch features event containing datapath information
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install the table-miss flow entry (default rule)
        match = parser.OFPMatch()  # Match all packets
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)  # Priority 0 (lowest)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle=0, hard=0):
        """
        Add a flow entry to the switch's flow table.
        
        Args:
            datapath: Switch to add the flow entry to
            priority: Priority of the flow entry
            match: Match conditions for the flow
            actions: Actions to perform on matching packets
            buffer_id: Buffer ID for the flow mod message
            idle: Idle timeout for the flow entry
            hard: Hard timeout for the flow entry
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Construct instruction to apply the specified actions
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
                                             
        # Create the flow mod message with or without buffer_id
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    idle_timeout=idle, hard_timeout=hard,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    idle_timeout=idle, hard_timeout=hard,
                                    match=match, instructions=inst)
            
        # Send the flow mod message to the switch
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
        Handler for PacketIn events - processes packets sent to the controller.
        Learns MAC addresses and installs flow entries for packet forwarding.
        
        Args:
            ev: PacketIn event with the packet data and metadata
        """
        # Check if the packet is truncated
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
                              
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']  # Port where packet was received

        # Parse packet and extract ethernet header
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
            
        dst = eth.dst  # Destination MAC address
        src = eth.src  # Source MAC address

        # Get datapath ID (switch ID)
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # Learn the MAC address to avoid FLOOD next time
        self.mac_to_port[dpid][src] = in_port

        # Determine output port based on destination MAC
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]  # Known destination
        else:
            out_port = ofproto.OFPP_FLOOD  # Unknown destination, flood

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid packet_in next time - only for known destinations
        if out_port != ofproto.OFPP_FLOOD:
            # Only for IP packets
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto

                # Create matches based on IP protocol type (ICMP, TCP, UDP)
                if protocol == in_proto.IPPROTO_ICMP:
                    t = pkt.get_protocol(icmp.icmp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol,icmpv4_code=t.code,
                                            icmpv4_type=t.type)
                elif protocol == in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol,
                                            tcp_src=t.src_port, tcp_dst=t.dst_port)
                elif protocol == in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip, ipv4_dst=dstip,
                                            ip_proto=protocol,
                                            udp_src=u.src_port, udp_dst=u.dst_port)

                # Install flow rule with timeout values (20s idle, 100s hard)
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle=20, hard=100)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, idle=20, hard=100)
                    
        # Send packet out if needed (when we don't have buffer_id)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
