import os
import sys
sys.path.append(os.getcwd())

from model import CNN
from flask import Flask, request, jsonify
from functools import wraps
import json
import numpy as np
import torch


app = Flask(__name__)


@app.route("/")
def hello():
	return "Fermat-Analytics Listener \n"


@app.route("/predict", methods=["POST"])
def predict():
	arr = request.form.get("arr")
	arr = json.loads(arr)
	arr = np.array(arr) / 255.0
	arr = torch.tensor(arr).to(torch.float32).unsqueeze(0)
	print("arr:", arr)
	print(type(arr), arr.shape, arr.dtype)

	model = CNN()
	model.load_state_dict(torch.load("./basic_cnn.pt"))
	out, _ = model(arr)
	pred = torch.max(out, 1)[1].data.squeeze()
	print()
	print()
	print()
	print()
	
	resp = jsonify(prediction=int(pred), success=True)


	

	return resp

