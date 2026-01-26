from pymavlink import mavutil
import sys

class Mavlink:
    # establishing mavlink connection
    def __init__(self):
        self.baud_rate = 57600 # change later
        self.connection_string = "udpin:localhost:14540" # change to UART connection string
        self.conn = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate) 
        self.conn.wait_heartbeat()

    # send the message once then stream continuously
    def send_message(self, message_id, rate): # mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION
        rate = int(1e6 / rate)

        message = self.conn.mav.command_long_encode(
                self.conn.target_system,  # Target system ID
                self.conn.target_component,  # Target component ID
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  # ID of command to send
                0,  # Confirmation
                message_id,  # param1: Message ID to be streamed
                rate,   # param2: Interval in microseconds
                0,       # param3 (unused)
                0,       # param4 (unused)
                0,       # param5 (unused)
                0,       # param6 (unused)
                0        # param7 (unused)
                )
            
        self.conn.mav.send(message)

    def stream_data(self, message): # mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION
        msg = self.conn.recv_match(type=message, blocking=True) # 'ATTITUDE_QUATERNION'

        if not msg:
            return None
        if msg.get_type() == "BAD_DATA":
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
            return None
        else:
            return msg



