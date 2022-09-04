import random
import socket
import time
import json
import pickle
import torch
from client import Client 
from config import *

class Worker:
    
    def __init__(self):
        data = { 'typ': 'join', 'content': ''}
        data = self.send(data)
        gpu_id = data['content']['gpu_id']
        data_path = data['content']['data_path']
        dataset_name = data['content']['dataset_name']
        model = data['content']['model']
        indices = data['content']['indices']
        total_epochs = data['content']['total_epochs']
        id = data['content']['total_epochs']
        self.client = Client(gpu_id, data_path, dataset_name, model, indices, total_epochs, id)

    def send(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            data = pickle.dumps(data)
            sock.sendall(data)
            data = self.receive(sock)

            return data

    def receive(self, sock):
        data = b''
        while True:
            packet = sock.recv(BUFF_SIZE)
            data += packet
            if len(packet) < BUFF_SIZE: break

        return pickle.loads(data)

    def train(self, cycle):
        self.client.train(cycle)
        data = { 'typ': 'model', 'content': self.client.id }
        self.send(data)

    def calculate(self):
        q = self.client.before_select()
        data = { 'typ': 'q', 'content': q }
        kl = self.send(data)
        sumq = self.client.select(kl)
        data = { 'typ': 'sumq', 'content': sumq }
        self.send(data)

if __name__ == '__main__':
    worker = Worker()
    for i in range(TOTAL_ROUNDS):
        worker.train(i)
        worker.calculate()
