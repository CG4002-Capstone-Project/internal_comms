import argparse
import concurrent
import json
import struct
import time
import traceback

import numpy as np
import torch
from bluepy import btle
from joblib import load

import dnn_utils
import svc_utils

parser = argparse.ArgumentParser(description="BLE")
parser.add_argument("--debug", default=False, help="debug mode")
parser.add_argument("--train", default=False, help="train mode")
parser.add_argument("--model_type", help="svc or dnn model")
parser.add_argument("--model_path", help="path to model")
parser.add_argument("--scaler_path", help="path to scaler")

args = parser.parse_args()
debug = args.debug
train = args.train
model_type = args.model_type
model_path = args.model_path
scaler_path = args.scaler_path
print(debug, model_type, model_path, scaler_path)

activities = ["dab", "gun", "elbow"]

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


class UUIDS:
    SERIAL_COMMS = btle.UUID("0000dfb1-0000-1000-8000-00805f9b34fb")


class Delegate(btle.DefaultDelegate):
    def __init__(self, params):
        btle.DefaultDelegate.__init__(self)

    def handleNotification(self, cHandle, data):

        for idx in range(len(beetle_addresses)):
            if global_delegate_obj[idx] == self:
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

                    print("starting to process the data")

                    packetUnpacked = False

                    laptop_receiving_timestamp = time.time()

                    try:
                        packetType = struct.unpack("<h", packet[:2])[0]
                        # time packet
                        if packetType == 0:
                            packet = struct.unpack("<hhLLhhL", packet)
                            packetUnpacked = True
                            print(packet)
                        # imu data packet
                        elif packetType > 0:
                            packet = struct.unpack("<hhhhhhhhhh", packet)
                            packetUnpacked = True
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

                        elif packet[0] > 0:
                            if verifychecksum(packet) is True:
                                try:
                                    yaw = float("{0:.2f}".format(packet[1] / 100))
                                    pitch = float("{0:.2f}".format(packet[2] / 100))
                                    roll = float("{0:.2f}".format(packet[3] / 100))
                                    accx = float("{0:.4f}".format(packet[4] / 8192))
                                    accy = float("{0:.4f}".format(packet[5] / 8192))
                                    accz = float("{0:.4f}".format(packet[6] / 8192))

                                    if idx == 0:
                                        if debug:
                                            raw_data[address].append(
                                                (yaw, pitch, roll, accx, accy, accz)
                                            )
                                        if train:
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
                                        if train:
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
                                        if train:
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

                except btle.BTLEDisconnectError:
                    global_beetle[beetle.addr] = 0
                    reestablish_connection(beetle)

                except Exception:
                    print(traceback.format_exc())
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
                        # total_connected_devices += 1
                        initHandshake(beetle)
                        print("Connected to %s" % (address))

                        return
        except Exception:
            print(traceback.format_exc())
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
            if disconnected_devices == 3:
                devices = scanner.scan(2)
                for d in devices:
                    if d.addr in beetle_addresses:
                        establish_connection(d.addr)

            else:
                print("reconnecting to %s" % (beetle.addr))
                try:
                    Peripheral(beetle.addr).disconnect()
                except Exception:
                    print(traceback.format_exc())

                establish_connection(beetle.addr)
                print("re-connected to %s" % (beetle.addr))
                return

        except:
            time.sleep(1)

    # total_connected_devices += 1


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
                    global_beetle[beetle.addr] = 0
                    reestablish_connection(beetle)

        except Exception:
            print(traceback.format_exc())
            global_beetle[beetle.addr] = 0
            reestablish_connection(beetle)


if __name__ == "__main__":
    # global variables
    beetle1 = "80:30:DC:E9:25:07"
    beetle2 = "34:B1:F7:D2:35:97"
    beetle3 = "34:B1:F7:D2:35:9D"

    beetle_addresses = [beetle2]
    # global total_connected_devices
    # total_connected_devices = 0

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

    # establish_connection(beetle1)
    establish_connection(beetle2)
    # establish_connection(beetle3)

    start_time = time.time()

    # start collecting data only after 10s passed
    counter = 0
    while True:
        elapsed_time = time.time() - start_time
        if int(elapsed_time) >= 10:
            break
        else:
            print(f"Waiting for {counter}s")
            time.sleep(1)
            counter += 1

    dance_move = None

    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as data_executor:
            {
                data_executor.submit(getDanceData, beetle): beetle
                for beetle in global_beetle
            }
            if debug:
                inputs = np.array(raw_data[beetle2])
                print(inputs.shape)
                print("Predicted dance move:", dance_move)
                n_readings = 90
                start_time_step = 30
                num_time_steps = 60
                if inputs.shape[0] >= n_readings:
                    # yaw pitch roll accx accy accz
                    inputs = inputs[start_time_step : start_time_step + num_time_steps]
                    inputs = np.array(
                        [
                            [
                                inputs[:, 0],
                                inputs[:, 1],
                                inputs[:, 2],
                                inputs[:, 3],
                                inputs[:, 4],
                                inputs[:, 5],
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
                    raw_data[beetle2] = list()
