import serial
import threading
import time

port = "/dev/ttyACM0"
baud = 9600
ser = serial.Serial(port = "/dev/ttyACM0", baudrate = 9600)

def buzzer_on():
    ser.write(b"on\n")

