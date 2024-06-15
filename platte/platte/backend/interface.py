#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Platte: This file offers a simple interface if one wishes to use the installed toolbox.
"""

import os
import torch
import numpy as np
from .image_processing import process
HERE_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(HERE_PATH, os.pardir))

class Interface:
    def __init__(self, pretrained_models_path='./models/', gpu=-1):
        self._gpu = 'cuda:' + str(gpu)
        from .pseudomodels import ModelManager
        print(HERE_PATH)
        print(PARENT_PATH)
        self._model_manager = ModelManager(pretrained_models_path, verbose=True, pretrained=True, gpu=self._gpu)
        selected_models = self._model_manager.get_matchings(self._c.experiment_parameter_map.get_val('models'))

        for selected_model in selected_models:
            if torch.cuda.is_available() and self._gpu != 'cuda:-1':
                print("Trying to move model " + selected_model.name + " to cuda!")
                self._model_manager.cuda(selected_model.name)
                self.memory_check("Position 1")
        print(self._model_manager._model_map)

    def memory_check(self, position=None):
        print(position)
        for i in range(8):
            print(torch.cuda.memory_reserved(i))
            print(torch.cuda.memory_allocated(i))
            print("")


    def run(self, model_name, img):
        print("Computing saliency")
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        print(img.shape[2])
        #Find model
        model = self._model_manager.get_matching(model_name)
        if torch.cuda.is_available() and self._gpu != 'cuda:-1':
            img = img.cuda(torch.device(self._gpu))

        prediction = model.predict(img)
        prediction = prediction.cpu().detach().numpy()[0, 0]
       
        return prediction

    def attention(self, model_name, img):
        return None

if __name__ == '__main__':
    print("TEST")
