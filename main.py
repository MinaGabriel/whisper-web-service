
import numpy as np
import sys 
from whisper_api import model
import os
from flask import Flask, request
from flask import jsonify
import sys

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root(): 
    return jsonify({'message': 'Hello, World!'})

@app.route("/audio", methods=["GET", "POST"])
def fun(): 
    if os.path.exists("media/infer.wav"):
        os.remove("media/infer.wav") 
    with open('media/infer.wav', mode='bx') as f:
        f.write(request.get_data())
    result = model.transcribe("media/infer.wav")   
    print(result["text"])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='10.50.117.30', port=8000, debug=True)