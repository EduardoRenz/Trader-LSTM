#%%
from flask import Flask,request 
from flask_cors import CORS,cross_origin
import json
import requests
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

@app.route('/quote/<ativo>',methods=['GET'])
def getMarketData(ativo):
    bot = requests.session()
    r = bot.post('http://webfeeder.cedrotech.com/SignIn?login=dudarenz&password=102030')
    dados = bot.get(f'http://webfeeder.cedrotech.com/services/quotes/quote/{ativo}')
    return dados.json()

print(__name__)

# %%
if __name__ == '__main__':
    app.run(debug=True, port=5001) 