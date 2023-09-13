import os
import pandas as pd

class Data(object):
    def __init__(self, config):
        self.file_name = config['data']['data_file_name']
        self.config = config

    def read(self):
        print(1)