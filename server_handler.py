import numpy as np
import random
import socketserver
import pickle
from server import Server
import time
from config import *

class Handler(socketserver.BaseRequestHandler):

    def setup(self):
        self.server = Server()
        self.client_nums = 0

    def handle(self):
        data = self.receive_data()
        typ = data['typ']
        if typ == "join":
            self.handle_join(data)
        elif typ == "model":
            self.handle_model(data)
        elif typ == 'q':
            self.handle_q(data)
        elif typ == "sumq":
            self.handle_sumq(data)
        else:
            print("Wrong type!!")


    def handle_join(self, data):
        #indices = np.array_split(range(5717), self.client_nums)
        print("start to handle join from " + self.client_address[0])
        train_info = {
            'gpu_id': self.server.gpu_ids[self.client_nums],
            'data_path': self.server.data_path,
            'dataset_name': self.server.dataset_name,
            'model': self.server.model,
            'indices': self.server.indices[self.client_nums],
            'total_epochs': self.server.total_epochs,
            'id': self.client_nums,
        }
        self.client_nums += 1
        data = { 'typ': 'start', 'content': train_info }
        self.send_data(data)
        print("finished handle join")
    
    def handle_model(self, data):
        print("start to handle model from " + self.client_address[0])
        self.server.aggregate(data['content'])
        print("finished handle model")
        self.send_data("ok")

    def handle_q(self, data):
        print("start to handle q from " + self.client_address[0])
        kl = self.server.calculate_kl(data['content'])
        data = { 'typ': "kl", 'content': kl }
        self.send_data(data)
        print("finished handle q")

    def handle_sumq(self, data):
        print("start to handle sumq from " + self.client_address[0])
        self.server.update_p(data['content'])
        print("finished handle sumq")
        self.send_data("ok")

    def receive_data(self):
        data = b''
        while True:
            packet = self.request.recv(BUFF_SIZE)
            data += packet
            if len(packet) < BUFF_SIZE: break

        return pickle.loads(data)

    def send_data(self, data):
        data = pickle.dumps(data)
        self.request.sendall(data)

if __name__ == "__main__":
    server = socketserver.TCPServer((HOST, PORT), Handler)

    server.serve_forever()
