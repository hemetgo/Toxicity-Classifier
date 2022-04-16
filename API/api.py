from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from ToxicityClassifier import predict
from ToxicityClassifier import clean_string

app = Flask(__name__)

@app.route('/')
def index():
    return "Wellcome to the toxic sentence API"


@app.route('/test', methods=['GET'])
def get():
    return jsonify({'Toxicity': 0})

@app.route('/classify', methods=['POST'])
def create():
    json_data = request.get_json(force=True)
    sentence = json_data['sentence']
    pred = predict(sentence)[0]
    return jsonify({'toxic': bool(pred),
                    'clean_string' : clean_string(sentence)})

if __name__ == "__main__":
    app.run(debug=True)