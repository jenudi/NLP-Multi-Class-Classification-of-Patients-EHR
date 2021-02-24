from flask import Flask, request, jsonify
import json
import requests

url="http://localhost:8080/"
api="word2vec_cluster"
url+=api

sentence="Patient states that he has not been feeling the same over the past few weeks. He has no desire to complete daily activities and does not want to get up out of bed.  Patient stays that he is always tired.  He can not think of any life changes that occurred prior to this change in behavior.  Nothing increases his energy levels or makes him feel better.  The patient complains of no pain.  His mother believes that he needs to have his medication switched to Celexa because that is what she takes for her depression. "
sentence_json=json.dumps({"sentence": sentence})
req=requests.post(url, json=sentence_json)

print(req.json())