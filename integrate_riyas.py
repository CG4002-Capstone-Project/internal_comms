# The client file that is used from the dancer's laptop to the Ultra96 Server
# This client will send the message as required
# The message format will be as defined

import os
import sys
import random
import time

import socket
import threading

import base64
# import numpy as np
# from tkinter import Label, Tk
# import pandas as pd
from Crypto.Cipher import AES
from Crypto import Random
from bluepy import btle

import concurrent
from concurrent import futures
import threading
import multiprocessing
import time
import numpy  # to count labels and store in dict
import operator  # to get most predicted label
import json
import random  # RNG in worst case
import struct
import os

# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
#ACTIONS = ['muscle', 'weightlifting', 'shoutout', 'dumbbells', 'tornado', 'facewipe', 'pacman', 'shootingstar']
# Week 10 test: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ['muscle', 'weightlifting', 'shoutout']
POSITIONS = ['1 2 3', '3 2 1', '2 3 1', '3 1 2', '1 3 2', '2 1 3']
LOG_DIR = os.path.join(os.path.dirname(__file__), 'evaluation_logs')
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 3 # position, 1 action, sync 
PORT_NUM = [int(os.environ['RIYAS']), 9092, 9093]

ENCRYPT_BLOCK_SIZE = 16


class Client(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, key):
        super(Client, self).__init__()

        # # setup moves
        # self.actions = ACTIONS
        # self.position = POSITIONS 
        # self.n_moves = len(ACTIONS) * NUM_MOVE_PER_ACTION

        # # the moves should be a random distribution
        # self.move_idxs = list(range(self.n_moves)) * NUM_MOVE_PER_ACTION
        # assert self.n_moves == len(self.actions) * NUM_MOVE_PER_ACTION
        # self.action = None
        # self.action_set_time = None

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.group_id = group_id
        self.key = key

        self.dancer_positions = ['1', '2', '3']

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)


        # print('Start connecting... server address: %s port: %s' % server_address, file=sys.stderr)
        print('Start connecting>>>>>>>>>>>>')
        self.socket.connect(server_address)
        print('Connected')

    #To encrypt the message, which is a string
    def encrypt_message(self, message):
        raw_message =  "#" + message
        # print("raw_message: "+raw_message)
        padded_raw_message = raw_message + ' '* (ENCRYPT_BLOCK_SIZE-(len(raw_message)%ENCRYPT_BLOCK_SIZE))
        # print("padded_raw_message: " + padded_raw_message)
        iv = Random.new().read(AES.block_size)
        secret_key = bytes(str(self.key), encoding="utf8")
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        encrypted_message = base64.b64encode(iv + cipher.encrypt(bytes(padded_raw_message, "utf8")))
        # print("encrypted_message: ", encrypted_message)
        return encrypted_message

    #To send the message to the sever
    def send_message(self, message):
        encrypted_message = self.encrypt_message(message)
        # print("Sending message:", encrypted_message)
        self.socket.sendall(encrypted_message)

    def receive_dancer_position(self):
        dancer_position = self.socket.recv(1024)
        msg = dancer_position.decode("utf8")
        return msg

    def receive_timestamp(self):
        timestamp = self.socket.recv(1024)
        msg = timestamp.decode("utf8")
        return msg

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()





class UUIDS:
    SERIAL_COMMS = btle.UUID("0000dfb1-0000-1000-8000-00805f9b34fb")


class ScanDelegate(btle.DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print ("Discovered device", dev.addr)
        elif isNewData:
            print ("Received new data from", dev.addr)

class Delegate(btle.DefaultDelegate):

    def __init__(self, params):
        btle.DefaultDelegate.__init__(self)

    def handleNotification(self, cHandle, data):
    
        for idx in range(len(beetle_addresses)):
            if global_delegate_obj[idx] == self:
                print("receiving data from %s" % (beetle_addresses[idx]))

                print ("length of packet is...")

                packet = data

                address = beetle_addresses[idx]

                size = len(packet)
                
                print (size)

                processData = False

                if (size < 20):
                    if buffer[address] != b'':
                        packet = buffer[address] + data
                        if (len(packet) == 20):
                            processData = True
                            buffer[address] = b''
                        else:
                            buffer[address] = packet
                            #print ("doing checksum verification")
                            #print ("checksum failedddddddddddd")
                            #print ("dropped the packet")
                    else:
                        buffer[address] = packet
                        

                if ((size == 20) or (processData == True)):

                    buffer[address] = b''

                    print("starting to process the data")
                
                    # packet = data

                    packetUnpacked = False

                    # address = beetle_addresses[idx]
                    #print (idx)
                    #print (beetle_addresses[idx])
                    #print (address)

                    laptop_receiving_timestamp = time.time() 

                    try:
                        packetType = struct.unpack("<h", packet[:2])[0]
                        print ("starting to unpack the data ")
                        #print ("here1")
                        #print (packetType)
                        # time packet
                        if packetType == 0:
                            #print ("here2a")
                            #print (packet)
                            packet = struct.unpack("<hhLLhhL", packet)
                            #print ("here2b")
                            packetUnpacked = True
                            print ("unpack is complete ")
                            print ("packet is")
                            print (packet)
                            #print ("unpack is complete ")
                        # imu data packet
                        elif packetType > 0:
                            packet = struct.unpack("<hhhhhhhhhh", packet)
                            packetUnpacked = True
                            print ("unpack is complete ")
                            print ("packet is")
                            print (packet)
                            #print ("unpack is complete ")
                    except Exception as e: 
                        print ("here3")
                        print (e)
                        print ("here3b")
                        pass

                    if packetUnpacked == True:
                        if packet[0] == 0:
                            if verifytimechecksum(packet) is True:
                            #print ('t5')
                                        
                                try:
                                    timestamp_dict[address].append(packet[2])
                                    timestamp_dict[address].append(packet[3])

                                except Exception as e:
                                                
                                    print ("exception 5")
                                    print (e)
                                    timestamp_dict[address].append(0)
                                    timestamp_dict[address].append(0)
                                            
                                timestamp_dict[address].append(laptop_receiving_timestamp)
                        
                                clocksync_flag_dict[address] = True
                                                
                                with open(r'offset.txt', 'a') as file:
                                    file.write(json.dumps(address) + ": ")
                                    file.write(json.dumps(timestamp_dict[address]) + "\n")
                                    file.close()
                        
                        elif packet[0] > 0:
                            if verifychecksum(packet) is True:
                                #print ("here10")
                                try:
                                    yaw = float("{0:.2f}".format(packet[1] / 100))
                                    pitch = float("{0:.2f}".format(packet[2] / 100))
                                    roll = float("{0:.2f}".format(packet[3] / 100))
                                    accx = float("{0:.4f}".format(packet[4] / 8192))
                                    accy = float("{0:.4f}".format(packet[5] / 8192))
                                    accz = float("{0:.4f}".format(packet[6] / 8192))

                                    if (idx == 0):

                                        with open(r'data1.txt', 'a') as file:
                                            print ("writing data values")
                                            file.write(json.dumps(yaw) + " " + json.dumps(pitch) + " " + json.dumps(roll) + " " + json.dumps(accx) + " " + json.dumps(accy) + " " + json.dumps(accz) + "\n")
                                            file.close()
                                            print ("writing is complete")
                                            my_client.send_message(str(dancer_id) + '|' + str(yaw) + '|' + str(pitch) + '|' + str(roll) + '|' + str(accx) + '|' + str(accy) + '|' + str(accz))

                                    elif (idx == 1):
                                    
                                        with open(r'data2.txt', 'a') as file:
                                            print ("writing data values")
                                            file.write(json.dumps(yaw) + " " + json.dumps(pitch) + " " + json.dumps(roll) + " " + json.dumps(accx) + " " + json.dumps(accy) + " " + json.dumps(accz) + "\n")
                                            file.close()
                                    elif (idx == 2):

                                        with open(r'data3.txt', 'a') as file:
                                            print ("writing data values")
                                            file.write(json.dumps(yaw) + " " + json.dumps(pitch) + " " + json.dumps(roll) + " " + json.dumps(accx) + " " + json.dumps(accy) + " " + json.dumps(accz) + "\n")
                                            file.close()

                                except Exception as e:
                                    print ("exception 2")
                                    print (e)
                            else:
                                print ("dropping packet")

                else:
                    pass

def verifytimechecksum(data):
    print ("doing checksum verification")
    result = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5]
    #priint ("result is")
    #print (result)
    #print (data[7])

    if result == data[6]:
        print ("checksum verification passed")
        return True
    else:
        print ("checksum faileddddddddddddddddddddd")
        return False

def verifychecksum(data):
    print ("doing checksum verification")
    result = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5] ^ data[6] ^ data[8] ^ data[9]
    #priint ("result is")
    #print (result)
    #print (data[7])

    if result == data[7]:
        print ("checksum verification passed")
        return True
    else:
        print ("checksum faileddddddddddddddddddddd")
        return False
            
def initHandshake(beetle):
    
    retries = 0
        
    for characteristic in beetle.getCharacteristics():
        if characteristic.uuid == UUIDS.SERIAL_COMMS:
            laptop_sending_timestamp = time.time() 
            timestamp_dict[beetle.addr].append(laptop_sending_timestamp)
            
            print("sending 'H' packet to %s" %(beetle.addr))
            
            characteristic.write(
                bytes('H', 'UTF-8'), withResponse=False)
                 
            while True:
                try:
                    if beetle.waitForNotifications(2):
                        if clocksync_flag_dict[beetle.addr] is True:
                            # function for time calibration
                            try:
                                clock_offset_tmp = calculate_clock_offset(timestamp_dict[beetle.addr])
                                tmp_value_list = []
                                if clock_offset_tmp is not None:
                                    tmp_value_list.append(clock_offset_tmp)
                                    clock_offset_dict[beetle.addr] = tmp_value_list
                            except Exception as e:
                                print(e)
                            
                            timestamp_dict[beetle.addr].clear()
                            print("beetle %s clock offset: %i" %(beetle.addr, clock_offset_dict[beetle.addr][-1]))
                            clocksync_flag_dict[beetle.addr] = False
                     
                            return
                        else:
                            continue
                    else:
                        while True:
                            print("Failed to receive timestamp, sending 'H' packet to %s" % (beetle.addr))
                            characteristic.write(
                                bytes('H', 'UTF-8'), withResponse=False)
                            break
                
                except btle.BTLEDisconnectError:
                    #total_connected_devices -= 1
                    global_beetle[beetle.addr] = 0
                    reestablish_connection(beetle)
                
                except Exception as e:
                    print ("exception 2")
                    print (e)
                    #total_connected_devices -= 1
                    global_beetle[beetle.addr] = 0
                    reestablish_connection(beetle)


def establish_connection(address):
    while True:
        try:
            for idx in range(len(beetle_addresses)):
                # for initial connections or when any beetle is disconnected
                if beetle_addresses[idx] == address:
                    if global_beetle[idx] != 0:  # do not reconnect if already connected
                        return
                    else:
                        print("connecting with %s" % (address))
                        # creates a Peripheral object and makes a connection to the device
                        beetle = btle.Peripheral(address)
                        global_beetle[idx] = beetle
                        # creates and initialises the object instance.
                        beetle_delegate = Delegate(address)
                        global_delegate_obj[idx] = beetle_delegate
                        # stores a reference to a “delegate” object, which is called when asynchronous events such as Bluetooth notifications occur.
                        beetle.withDelegate(beetle_delegate)
                        #total_connected_devices += 1
                        initHandshake(beetle)
                        print("Connected to %s" % (address))

                        return
        except Exception as e:
            print ("exception 1")
            print(e)
            for idx in range(len(beetle_addresses)):
                # for initial connections or when any beetle is disconnected
                if beetle_addresses[idx] == address:
                    if global_beetle[idx] != 0:  # do not reconnect if already connected
                        return


def reestablish_connection(beetle):
    
    disconnected_devices = 0

    for idx in range(len(global_beetle)):
        if global_beetle[idx] == 0:
            disconnected_devices += 1

            
    while True:
        
        try:
            if (disconnected_devices == 3):
                devices = scanner.scan(2)
                for d in devices:
                    if d.addr in beetle_addresses:
                        establish_connection(d.addr)


            else:
                print("reconnecting to %s" % (beetle.addr))
                
                #Peripheral(self.address)._stopHelper()
                try:
                    Peripheral(beetle.addr).disconnect()
                except Exception as e:
                    print (e)

                establish_connection(beetle.addr)
                #beetle.connect(beetle.addr)
                print("re-connected to %s" % (beetle.addr))
                return
    
        except:
            time.sleep(1)
    

    #total_connected_devices += 1

def calculate_clock_offset(beetle_timestamp_list):
    print ('calculating Offset')
    if(len(beetle_timestamp_list) == 4) :
        RTT = (beetle_timestamp_list[3] - beetle_timestamp_list[0]) \
              - (beetle_timestamp_list[2] - beetle_timestamp_list[1])
        clock_offset = (beetle_timestamp_list[1] - beetle_timestamp_list[0]) - RTT/2
        # print (beetle_timestamp_list[3])
        # print (beetle_timestamp_list[2])
        # print (beetle_timestamp_list[1])
        # print (beetle_timestamp_list[0])
        
        # print ("clock_offset is ")
        # print(clock_offset)
        return clock_offset
    else:
        print("error in beetle timestamp")
        print(len(beetle_timestamp_list))
        print((beetle_timestamp_list))
        return None

def getDanceData(beetle):
    
    print ("time for imu data")
    #print ("1")

    #for characteristic in beetle.getCharacteristics():
    #    print ("2")

    #    if characteristic.uuid == UUIDS.SERIAL_COMMS:
            #laptop_sending_timestamp = 1
            #timestamp_dict[beetle.addr].append(laptop_sending_timestamp)
            # characteristic.write(bytes('H', 'UTF-8'), withResponse=False)
    #        print ("char sent")
            
    waitCount = 0

    #print ("3")

    while True:
        
        try:
            if beetle.waitForNotifications(2):
                # processData("34:B1:F7:D2:35:9D")
                print("collected imu data")
                return
            
            else:
                waitCount += 1
                if (waitCount >= 10):
                    waitCount = 0
                    #total_connected_devices -= 1
                    #print ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    global_beetle[beetle.addr] = 0
                    reestablish_connection(beetle)

        #except btle.BTLEDisconnectError:
            #total_connected_devices -= 1
        #    reestablish_connection(beetle)

        except Exception as e:
            print ("exception 3")
            print (e)
            #total_connected_devices -= 1
            global_beetle[beetle.addr] = 0
            reestablish_connection(beetle)

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('Invalid number of arguments')
        print('python server.py [dancer ID]')
        sys.exit()
    
    
    dancer_id = int(sys.argv[1])
    #dancer_id = 1
    ip_addr = "127.0.0.1"
    #sys.argv[1]
    port_num = PORT_NUM[dancer_id]
    group_id = "18"#sys.argv[3]
    key = "1234123412341234"#sys.argv[4]

    my_client = Client(ip_addr, port_num, group_id, key)

    index = 0

    time.sleep(5)

    RTT = 0.0
    offset = 0.0

    
    my_client.send_message(str(dancer_id) + '|' + str(RTT) + '|' + str(offset) + '|' )

    # global variables
   
    beetle_addresses = ["34:B1:F7:D2:35:9D"]
    #global total_connected_devices
    #total_connected_devices = 0

    divide_ypr = 100
    divide_acc = 8192

    global_delegate_obj = []
    global_beetle = []

    # used to notify if sync data is available
    clocksync_flag_dict = {"80:30:DC:E9:25:07": False, "34:B1:F7:D2:35:97": False, "34:B1:F7:D2:35:9D": False}
    # used to hold laptop and beetle receive and send time
    timestamp_dict = {"80:30:DC:E9:25:07": [], "34:B1:F7:D2:35:97": [], "34:B1:F7:D2:35:9D": []}
    # used to hold clock offset values
    clock_offset_dict = {"80:30:DC:E9:25:07": [], "34:B1:F7:D2:35:97": [], "34:B1:F7:D2:35:9D": []}
    buffer = {"80:30:DC:E9:25:07": b'', "34:B1:F7:D2:35:97": b'', "34:B1:F7:D2:35:9D": b''}

    [global_delegate_obj.append(0) for idx in range(len(beetle_addresses))]
    [global_beetle.append(0) for idx in range(len(beetle_addresses))]

   
    #with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #    for bluno in beetle_addresses:
    #        executor.submit(establish_connection, "34:B1:F7:D2:35:9D")
    
    beetle1 = "80:30:DC:E9:25:07"
    beetle2 = "34:B1:F7:D2:35:97"
    beetle3 = "34:B1:F7:D2:35:9D"
    
    #establish_connection(beetle1)
    #establish_connection(beetle2)
    establish_connection(beetle3)

    start_time = time.time()

    # start collecting data only after 10s passed
    while True:
        elapsed_time = time.time() - start_time
        if int(elapsed_time) >= 10:
            break
        else:
            print ("Waiting for 10s")
            time.sleep(1)
      
    print ("10s done")

    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as data_executor:
            print ("getting data from beetles")
            {data_executor.submit(getDanceData, beetle): beetle for beetle in global_beetle}
            #data_executor.shutdown(wait=True)

    """
    while True:
        t1 = time.time()
        print("t1: " + str(t1))
        my_client.send_message(str(dancer_id) + '|' + str(RTT) + '|' + str(offset) + '|' )
        timestamp = my_client.receive_timestamp()
        t4 = time.time()
        print("t4: " + str(t4))
        t2 = float(timestamp.split("|")[0])
        print("t2: " + str(t2))
        t3 = float(timestamp.split("|")[1])
        print("t3: " + str(t3))

        RTT = (t4 -t3 + t2 - t1)
        print("RTT: " + str(RTT))
        offset = (t2 - t1) - RTT/2
        print("offset: " + str(offset))

        time.sleep(2)
    """
        


#if __name__ == '__main__':
    #main()