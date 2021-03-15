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
import pika

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
PORT_NUM = [9091, 9092, 9093]


BUFFER = [] #The buffer to store the message from the beetles

ENCRYPT_BLOCK_SIZE = 16

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
                        elif packetType == 5:
                            packet = struct.unpack("<hLLLhL", packet)
                            packetUnpacked = True
                            print ("unpack is complete ")
                            print ("packet is")
                            print (packet)
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
                                                
                                with open(r'offset1.txt', 'a') as file:
                                    file.write(json.dumps(address) + ": ")
                                    file.write(json.dumps(timestamp_dict[address]) + "\n")
                                    file.close()
                        
                        elif packet[0] == 5:
                            if verifyemgchecksum(packet) is True:
                                #print ("here10")
                                try:

                                    mav = float("{0:.2f}".format(packet[1] / 100))
                                    rms = float("{0:.2f}".format(packet[2] / 100))
                                    meanfreq = float("{0:.2f}".format(packet[3] / 100))
                                    

                                    BUFFER.append(str(mav) + " " + str(rms) + " " + str(meanfreq))

                                    if (idx == 0):

                                        with open(r'emg1.txt', 'a') as file:
                                            print ("writing data values")
                                            file.write(json.dumps(mav) + " " + json.dumps(rms) + " " + json.dumps(meanfreq) + "\n")
                                            file.close()
                                            print ("writing is complete")
                                            # my_client.send_message()

                                except Exception as e:
                                    print ("exception 1")
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

def verifyemgchecksum(data):
    print ("doing checksum verification")
    result = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] 
    #priint ("result is")
    #print (result)
    #print (data[7])

    if result == data[5]:
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
                except KeyboardInterrupt:
                    for idx in range(len(global_beetle)):
                        if global_beetle[idx] != 0:  # disconnect before reconnect
                      
                            global_beetle[idx]._stopHelper()
                   
                            global_beetle[idx].disconnect()
                            
                            global_beetle[idx] = 0


                except Exception as e:
                    print ("exception 2")
                    print (e)
                    establish_connection(beetle.addr)
                    return


def establish_connection(address):
    while True:
        try:
            for idx in range(len(beetle_addresses)):
                # for initial connections or when any beetle is disconnected
                if beetle_addresses[idx] == address:
                    if global_beetle[idx] != 0:  # disconnect before reconnect
                        global_beetle[idx]._stopHelper()
                        global_beetle[idx].disconnect()
                        global_beetle[idx] = 0
                    if global_beetle[idx] == 0: # just stick with if instead of else
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
            establish_connection(address)
            return
                


def calculate_clock_offset(beetle_timestamp_list):
    print ('calculating Offset')
    if(len(beetle_timestamp_list) == 4) :
        RTT = (beetle_timestamp_list[3] - beetle_timestamp_list[0]) \
              - (beetle_timestamp_list[2] - beetle_timestamp_list[1])
        clock_offset = (beetle_timestamp_list[1] - beetle_timestamp_list[0]) - RTT/2
        return clock_offset
    else:
        print("error in beetle timestamp")
        print(len(beetle_timestamp_list))
        print((beetle_timestamp_list))
        return None

def getDanceData(beetle):
    
    print ("time for imu data")
            
    waitCount = 0

    while True:
        
        try:
            if beetle.waitForNotifications(2):
                print("collected imu data")
                return
            
            else:
                waitCount += 1
                if (waitCount >= 10): # no data for approx 10 seconds and so reconnect
                    waitCount = 0
                    establish_connection(beetle.addr)
                    retrun

        except Exception as e:
            print ("exception 3")
            print (e)
            establish_connection(beetle.addr)
            return
            
       

if __name__ == '__main__':

    try:

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
        
        # my_client.send_message(str(dancer_id) + '|' + str(RTT) + '|' + str(offset) + '|' )

        # global variables

        beetle_addresses = ["34:14:B5:51:D1:6B"]
        #global total_connected_devices
        #total_connected_devices = 0

        divide_ypr = 100
        divide_acc = 8192

        global_delegate_obj = []
        global_beetle = []

        # used to notify if sync data is available
        clocksync_flag_dict = {"80:30:DC:E9:25:07": False, "34:B1:F7:D2:35:97": False, "34:B1:F7:D2:35:9D": False, "34:14:B5:51:D1:6B": False} 
        # used to hold laptop and beetle receive and send time
        timestamp_dict = {"80:30:DC:E9:25:07": [], "34:B1:F7:D2:35:97": [], "34:B1:F7:D2:35:9D": [], "34:14:B5:51:D1:6B": []}
        # used to hold clock offset values
        clock_offset_dict = {"80:30:DC:E9:25:07": [], "34:B1:F7:D2:35:97": [], "34:B1:F7:D2:35:9D": [], "34:14:B5:51:D1:6B": []}
        buffer = {"80:30:DC:E9:25:07": b'', "34:B1:F7:D2:35:97": b'', "34:B1:F7:D2:35:9D": b'', "34:14:B5:51:D1:6B": b''}

        [global_delegate_obj.append(0) for idx in range(len(beetle_addresses))]
        [global_beetle.append(0) for idx in range(len(beetle_addresses))]


        beetle1 = "80:30:DC:E9:25:07"
        beetle2 = "34:B1:F7:D2:35:97"
        beetle3 = "34:B1:F7:D2:35:9D"
        beetle4 = "34:14:B5:51:D1:6B"
        
        #establish_connection(beetle1)
        #establish_connection(beetle2)
        establish_connection(beetle4)

        start_time = time.time()

        
        #establish the client connection with Ultra96 Server
        #my_client = Client(ip_addr, port_num, group_id, key)

        index = 0

        print("waiting for 10s")
        time.sleep(10)

        RTT = 0.0
        offset = 0.0

        # Parse CLODUAMQP_URL (fallback to localhost)
        params = pika.URLParameters("amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu")
        params.socket_timeout = 5

        connection = pika.BlockingConnection(params) # Connect to CloudAMQP
        channel = connection.channel() # start a channel

        channel.queue_declare(queue='emg') # Declare a queue


        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as data_executor:
                print ("getting data from beetles")
                {data_executor.submit(getDanceData, beetle): beetle for beetle in global_beetle}
                #data_executor.shutdown(wait=True)

                if len(BUFFER) > 0:
                    database_emg_data = BUFFER.pop(0)
                    print("data for DB...")
                    print(database_emg_data)
                    t1 = time.time()
                    emg_data = (
                    
                        str(t1)
                        + "|"
                        + database_emg_data
                        + "|"
                    )
                    print (emg_data)
                    channel.basic_publish(exchange='', routing_key='emg', body=emg_data)


                
    except KeyboardInterrupt:
        for idx in range(len(global_beetle)):
            if global_beetle[idx] != 0:  # disconnect before reconnect
                print ("a1")
                global_beetle[idx]._stopHelper()
                print ("a2")
                global_beetle[idx].disconnect()
                print ("a3")
                global_beetle[idx] = 0

