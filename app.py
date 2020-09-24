#%%
from flask import Flask,request 
from flask_cors import CORS,cross_origin
import json
app = Flask(__name__)
CORS(app)

@app.route('/negocios',methods=['GET', 'POST'])
def hello_world():
    print(request.data )
    return 'hi'

@app.route('/trades',methods=['POST'])
@cross_origin()
def saveTrades():
    data = json.loads(request.data)
    with open(f'inputs/{data["quoteTrade"]["M"]}_trades.json', 'a+') as outfile:
        offers = data["L"]
        for offer in offers:
            print(offer)
            outfile.write(json.dumps(offer)+",")
    return 'ok'


print(__name__)

# %%
if __name__ == '__main__':
    app.run(debug=True, port=5001) 