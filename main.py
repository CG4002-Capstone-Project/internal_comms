import argparse
import base64
import concurrent
import json
import socket
import struct
import sys
import threading
import time
import traceback

import numpy as np
import pika
import torch
from bluepy import btle
from Crypto import Random
from Crypto.Cipher import AES
from joblib import load

import dnn_utils
import svc_utils

PORT_NUM = [9091, 9092, 9093]


BUFFER = []  # The buffer to store the message from the beetles

ENCRYPT_BLOCK_SIZE = 16


class Client(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, key):
        super(Client, self).__init__()

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.group_id = group_id
        self.key = key

        self.dancer_positions = ["1", "2", "3"]

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        # print('Start connecting... server address: %s port: %s' % server_address, file=sys.stderr)
        print("Start connecting>>>>>>>>>>>>")
        self.socket.connect(server_address)
        print("Connected")

    # To encrypt the message, which is a string
    def encrypt_message(self, message):
        raw_message = "#" + message
        # print("raw_message: "+raw_message)
        padded_raw_message = raw_message + " " * (
            ENCRYPT_BLOCK_SIZE - (len(raw_message) % ENCRYPT_BLOCK_SIZE)
        )
        # print("padded_raw_message: " + padded_raw_message)
        iv = Random.new().read(AES.block_size)
        secret_key = bytes(str(self.key), encoding="utf8")
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        encrypted_message = base64.b64encode(
            iv + cipher.encrypt(bytes(padded_raw_message, "utf8"))
        )
        print("encrypted_message: ", encrypted_message)
        return encrypted_message

    # To send the message to the sever
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


class Delegate(btle.DefaultDelegate):
    def __init__(self, params):
        btle.DefaultDelegate.__init__(self)

    def handleNotification(self, cHandle, data):

        for idx in range(len(beetle_addresses)):
            if global_delegate_obj[idx] == self:
                if verbose:
                    print("receiving data from %s" % (beetle_addresses[idx]))
                packet = data
                address = beetle_addresses[idx]
                size = len(packet)
                processData = False

                if size < 20:
                    if buffer[address] != b"":
                        packet = buffer[address] + data
                        if len(packet) == 20:
                            processData = True
                            buffer[address] = b""
                        else:
                            buffer[address] = packet
                    else:
                        buffer[address] = packet

                if (size == 20) or (processData == True):

                    buffer[address] = b""

                    packetUnpacked = False

                    laptop_receiving_timestamp = time.time()

                    try:
                        packetType = struct.unpack("<h", packet[:2])[0]
                        # time packet
                        if packetType == 0:
                            packet = struct.unpack("<hhLLhhL", packet)
                            packetUnpacked = True
                            if verbose:
                                print(packet)
                        # imu data packet
                        elif packetType > 0:
                            packet = struct.unpack("<hhhhhhhhhh", packet)
                            packetUnpacked = True
                            if verbose:
                                print(packet)
                    except Exception:
                        print(traceback.format_exc())

                    if packetUnpacked == True:
                        if packet[0] == 0:
                            if verifytimechecksum(packet) is True:
                                try:
                                    timestamp_dict[address].append(packet[2])
                                    timestamp_dict[address].append(packet[3])

                                except Exception:
                                    print(traceback.format_exc())
                                    timestamp_dict[address].append(0)
                                    timestamp_dict[address].append(0)

                                timestamp_dict[address].append(
                                    laptop_receiving_timestamp
                                )

                                clocksync_flag_dict[address] = True

                                with open(r"offset.txt", "a") as file:
                                    file.write(json.dumps(address) + ": ")
                                    file.write(
                                        json.dumps(timestamp_dict[address]) + "\n"
                                    )
                                    file.close()

                        elif (packet[0] > 0) and (allow_imu[address] == True):
                            if verifychecksum(packet) is True:
                                try:
                                    mode = packet[0]
                                    yaw = float("{0:.2f}".format(packet[1] / 100))
                                    pitch = float("{0:.2f}".format(packet[2] / 100))
                                    roll = float("{0:.2f}".format(packet[3] / 100))
                                    accx = float("{0:.4f}".format(packet[4] / 8192))
                                    accy = float("{0:.4f}".format(packet[5] / 8192))
                                    accz = float("{0:.4f}".format(packet[6] / 8192))

                                    if idx == 0:
                                        if production:
                                            BUFFER.append(
                                                str(mode)
                                                + " "
                                                + str(yaw)
                                                + " "
                                                + str(pitch)
                                                + " "
                                                + str(roll)
                                                + " "
                                                + str(accx)
                                                + " "
                                                + str(accy)
                                                + " "
                                                + str(accz)
                                            )
                                        if debug:
                                            raw_data[address].append(
                                                (
                                                    mode,
                                                    yaw,
                                                    pitch,
                                                    roll,
                                                    accx,
                                                    accy,
                                                    accz,
                                                )
                                            )
                                        if collect:
                                            with open(r"data1.txt", "a") as file:
                                                print("writing data values")
                                                file.write(
                                                    json.dumps(yaw)
                                                    + " "
                                                    + json.dumps(pitch)
                                                    + " "
                                                    + json.dumps(roll)
                                                    + " "
                                                    + json.dumps(accx)
                                                    + " "
                                                    + json.dumps(accy)
                                                    + " "
                                                    + json.dumps(accz)
                                                    + "\n"
                                                )
                                                file.close()
                                                print("writing is complete")

                                    elif idx == 1:
                                        if collect:
                                            with open(r"data2.txt", "a") as file:
                                                print("writing data values")
                                                file.write(
                                                    json.dumps(yaw)
                                                    + " "
                                                    + json.dumps(pitch)
                                                    + " "
                                                    + json.dumps(roll)
                                                    + " "
                                                    + json.dumps(accx)
                                                    + " "
                                                    + json.dumps(accy)
                                                    + " "
                                                    + json.dumps(accz)
                                                    + "\n"
                                                )
                                                file.close()
                                    elif idx == 2:
                                        if collect:
                                            with open(r"data3.txt", "a") as file:
                                                print("writing data values")
                                                file.write(
                                                    json.dumps(yaw)
                                                    + " "
                                                    + json.dumps(pitch)
                                                    + " "
                                                    + json.dumps(roll)
                                                    + " "
                                                    + json.dumps(accx)
                                                    + " "
                                                    + json.dumps(accy)
                                                    + " "
                                                    + json.dumps(accz)
                                                    + "\n"
                                                )
                                                file.close()

                                except Exception:
                                    print(traceback.format_exc())
                            else:
                                print("dropping packet")

                else:
                    pass


def verifytimechecksum(data):
    result = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5]
    if result == data[6]:
        print("checksum verification passed")
        return True
    else:
        print("checksum failed")
        return False


def verifychecksum(data):
    result = (
        data[0]
        ^ data[1]
        ^ data[2]
        ^ data[3]
        ^ data[4]
        ^ data[5]
        ^ data[6]
        ^ data[8]
        ^ data[9]
    )
    if result == data[7]:
        if verbose:
            print("checksum verification passed")
        return True
    else:
        print("checksum failed")
        return False


def initHandshake(beetle):
    for characteristic in beetle.getCharacteristics():
        if characteristic.uuid == UUIDS.SERIAL_COMMS:
            laptop_sending_timestamp = time.time()
            timestamp_dict[beetle.addr].append(laptop_sending_timestamp)

            print("sending 'H' packet to %s" % (beetle.addr))

            characteristic.write(bytes("H", "UTF-8"), withResponse=False)

            while True:
                try:
                    if beetle.waitForNotifications(2):
                        if clocksync_flag_dict[beetle.addr] is True:
                            # function for time calibration
                            try:
                                clock_offset_tmp = calculate_clock_offset(
                                    timestamp_dict[beetle.addr]
                                )
                                tmp_value_list = []
                                if clock_offset_tmp is not None:
                                    tmp_value_list.append(clock_offset_tmp)
                                    clock_offset_dict[beetle.addr] = tmp_value_list
                            except Exception:
                                print(traceback.format_exc())
                            timestamp_dict[beetle.addr].clear()
                            print(
                                "beetle %s clock offset: %i"
                                % (beetle.addr, clock_offset_dict[beetle.addr][-1])
                            )
                            clocksync_flag_dict[beetle.addr] = False

                            return
                        else:
                            continue
                    else:
                        while True:
                            print(
                                "Failed to receive timestamp, sending 'H' packet to %s"
                                % (beetle.addr)
                            )
                            characteristic.write(
                                bytes("H", "UTF-8"), withResponse=False
                            )
                            break
                except KeyboardInterrupt:
                    print(traceback.format_exc())
                    if global_beetle[0] != 0:  # disconnect
                        global_beetle[0]._stopHelper()
                        global_beetle[0].disconnect()
                        global_beetle[0] = 0
                    sys.exit()
                except Exception:
                    print(traceback.format_exc())
                    establish_connection(beetle.addr)
                    return


def establish_connection(address):
    allow_imu[address] = False
    # BUFFER = [] TODO: Riyas
    while True:
        try:
            for idx in range(len(beetle_addresses)):
                # for initial connections or when any beetle is disconnected
                if beetle_addresses[idx] == address:
                    if global_beetle[idx] != 0:  # disconnect before reconnect
                        global_beetle[idx]._stopHelper()
                        global_beetle[idx].disconnect()
                        global_beetle[idx] = 0
                    if global_beetle[idx] == 0:  # just stick with if instead of else
                        print("connecting with %s" % (address))
                        # creates a Peripheral object and makes a connection to the device
                        beetle = btle.Peripheral(address)
                        global_beetle[idx] = beetle
                        # creates and initialises the object instance.
                        beetle_delegate = Delegate(address)
                        global_delegate_obj[idx] = beetle_delegate
                        # stores a reference to a “delegate” object, which is called when asynchronous events such as Bluetooth notifications occur.
                        beetle.withDelegate(beetle_delegate)
                        # total_connected_devices += 1
                        initHandshake(beetle)
                        print("Connected to %s" % (address))
                        allow_imu[address] = True
                        return
        except KeyboardInterrupt:
            print(traceback.format_exc())
            if global_beetle[0] != 0:  # disconnect
                global_beetle[0]._stopHelper()
                global_beetle[0].disconnect()
                global_beetle[0] = 0
            sys.exit()
        except Exception:
            print(traceback.format_exc())
            establish_connection(address)
            return


def calculate_clock_offset(beetle_timestamp_list):
    print("calculating Offset")
    if len(beetle_timestamp_list) == 4:
        RTT = (beetle_timestamp_list[3] - beetle_timestamp_list[0]) - (
            beetle_timestamp_list[2] - beetle_timestamp_list[1]
        )
        clock_offset = (beetle_timestamp_list[1] - beetle_timestamp_list[0]) - RTT / 2
        return clock_offset
    else:
        print("error in beetle timestamp")
        print(len(beetle_timestamp_list))
        print((beetle_timestamp_list))
        return None


def getDanceData(beetle):
    waitCount = 0
    while True:
        try:
            if beetle.waitForNotifications(2):
                return
            else:
                waitCount += 1
                if waitCount >= 10:
                    waitCount = 0
                    establish_connection(beetle.addr)
                    return

        except KeyboardInterrupt:
            print(traceback.format_exc())
            if global_beetle[0] != 0:  # disconnect
                global_beetle[0]._stopHelper()
                global_beetle[0].disconnect()
                global_beetle[0] = 0
            sys.exit()
        except Exception:
            print(traceback.format_exc())
            establish_connection(beetle.addr)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument("--beetle_id", help="beetle id", type=int, required=True)
    parser.add_argument("--dancer_id", help="dancer id", type=int, required=True)
    parser.add_argument("--debug", default=False, help="debug mode", type=bool)
    parser.add_argument("--collect", default=False, help="train mode", type=bool)
    parser.add_argument(
        "--production", default=False, help="production mode", type=bool
    )
    parser.add_argument(
        "--dashboard", default=False, help="send to dashboard", type=bool
    )
    parser.add_argument("--ultra96", default=False, help="send to ultra96", type=bool)
    parser.add_argument("--verbose", default=False, help="verbose", type=bool)
    parser.add_argument("--model_type", help="svc or dnn model")
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--scaler_path", help="path to scaler")
    parser.add_argument(
        "--cloudamqp_url",
        default="amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu",
        help="dashboard connection",
    )

    args = parser.parse_args()
    beetle_id = args.beetle_id
    dancer_id = args.dancer_id
    debug = args.debug
    collect = args.collect
    production = args.production
    dashboard = args.dashboard
    ultra96 = args.ultra96
    verbose = args.verbose
    model_type = args.model_type
    model_path = args.model_path
    scaler_path = args.scaler_path
    cloudamqp_url = args.cloudamqp_url

    print("beetle_id:", beetle_id)
    print("dancer_id:", dancer_id)
    print("debug:", debug)
    print("collect:", collect)
    print("production:", production)
    print("verbose:", verbose)
    print("model_type:", model_type)
    print("model_path:", model_path)
    print("scaler_path:", scaler_path)
    print("cloudamqp_url:", cloudamqp_url)

    ip_addr = "127.0.0.1"
    port_num = PORT_NUM[dancer_id]
    group_id = "18"
    key = "1234123412341234"
    activities = ["gun", "sidepump", "hair"]

    if debug:
        # Load scaler
        scaler = load(scaler_path)

        # Load model
        if model_type == "svc":
            model = load(model_path)
        elif model_type == "dnn":
            model = dnn_utils.DNN()
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            raise Exception("Model is not supported")

    # global variables
    beetle1 = "80:30:DC:E9:25:07"
    beetle2 = "34:B1:F7:D2:35:97"
    beetle3 = "34:B1:F7:D2:35:9D"

    beetle_addresses = list()
    if beetle_id == 1:
        beetle_addresses.append(beetle1)
    if beetle_id == 2:
        beetle_addresses.append(beetle2)
    if beetle_id == 3:
        beetle_addresses.append(beetle3)

    divide_ypr = 100
    divide_acc = 8192

    global_delegate_obj = []
    global_beetle = []

    # used to notify if sync data is available
    clocksync_flag_dict = {
        beetle1: False,
        beetle2: False,
        beetle3: False,
    }
    allow_imu = {
        beetle1: False,
        beetle2: False,
        beetle3: False,
    }
    # used to hold laptop and beetle receive and send time
    timestamp_dict = {
        beetle1: [],
        beetle2: [],
        beetle3: [],
    }
    # used to hold clock offset values
    clock_offset_dict = {
        beetle1: [],
        beetle2: [],
        beetle3: [],
    }
    buffer = {
        beetle1: b"",
        beetle2: b"",
        beetle3: b"",
    }

    [global_delegate_obj.append(0) for idx in range(len(beetle_addresses))]
    [global_beetle.append(0) for idx in range(len(beetle_addresses))]

    raw_data = {
        beetle1: [],
        beetle2: [],
        beetle3: [],
    }

    if beetle_id == 1:
        establish_connection(beetle1)
    if beetle_id == 2:
        establish_connection(beetle2)
    if beetle_id == 3:
        establish_connection(beetle3)

    start_time = time.time()

    if production and ultra96:
        my_client = Client(ip_addr, port_num, group_id, key)

    # Parse CLODUAMQP_URL (fallback to localhost)
    params = pika.URLParameters(cloudamqp_url)
    params.socket_timeout = 5

    connection = pika.BlockingConnection(params)  # Connect to CloudAMQP
    channel = connection.channel()  # start a channel

    channel.queue_declare(queue="raw_data")  # Declare a queue

    print("waiting for 10s")
    time.sleep(10)
    print("start")

    is_init = True
    RTT = 0.0
    offset = 0.0
    target_beetle = beetle_addresses[0]  # TODO: Change after Week 9
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as data_executor:
            {
                data_executor.submit(getDanceData, beetle): beetle
                for beetle in global_beetle
            }
            if is_init:
                raw_data[target_beetle] = list()
                is_init = False
                continue
            if production:
                print(len(BUFFER))
                if len(BUFFER) >= 8:
                    # current_data = BUFFER.pop(0)
                    current_data = "#".join(BUFFER)
                    BUFFER = list()
                    t1 = time.time()
                    if dashboard:
                        database_msg = (
                            str(dancer_id) + "|" + str(t1) + "|" + current_data + "|"
                        )
                        channel.basic_publish(
                            exchange="", routing_key="raw_data", body=database_msg
                        )
                        if verbose:
                            print(database_msg)
                    if ultra96:
                        message_final = (
                            str(dancer_id)
                            + "|"
                            + str(RTT)
                            + "|"
                            + str(offset)
                            + "|"
                            + current_data
                            + "|"
                        )
                        if verbose:
                            print("current_data: " + current_data)
                            print("message_final: " + message_final)

                        my_client.send_message(message_final)
                        timestamp = my_client.receive_timestamp()
                        t4 = time.time()
                        t2 = float(timestamp.split("|")[0][:18])
                        t3 = float(timestamp.split("|")[1][:18])
                        RTT = t4 - t3 + t2 - t1
                        offset = (t2 - t1) - RTT / 2

            if debug:
                inputs = np.array(raw_data[target_beetle])
                n_readings = 90
                start_time_step = 30
                num_time_steps = 60
                if inputs.shape[0] >= n_readings:
                    # yaw pitch roll accx accy accz
                    inputs = inputs[start_time_step : start_time_step + num_time_steps]
                    inputs = np.array(
                        [
                            [
                                inputs[:, 1],
                                inputs[:, 2],
                                inputs[:, 3],
                                inputs[:, 4],
                                inputs[:, 5],
                                inputs[:, 6],
                            ]
                        ]
                    )
                    if model_type == "svc":
                        inputs = svc_utils.extract_raw_data_features(
                            inputs
                        )  # extract features
                        inputs = svc_utils.scale_data(inputs, scaler)  # scale features
                        predicted = model.predict(inputs)[0]
                        dance_move = activities[predicted]
                    elif model_type == "dnn":
                        inputs = dnn_utils.extract_raw_data_features(
                            inputs
                        )  # extract features
                        inputs = dnn_utils.scale_data(inputs, scaler)  # scale features
                        inputs = torch.tensor(inputs)  # convert to tensor
                        outputs = model(inputs.float())
                        _, predicted = torch.max(outputs.data, 1)
                        dance_move = activities[predicted]
                    else:
                        raise Exception("Model is not supported")
                    raw_data[target_beetle] = list()
                    print("Predicted:", dance_move)
