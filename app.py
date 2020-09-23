#%%
from flask import Flask,request 
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app)

@app.route('/negocios',methods=['GET', 'POST'])
def hello_world():
    print(request.data )
    return 'hi'

@app.route('/trades',methods=['POST'])
@cross_origin()
def saveTrades():
    print(request.data )
    return 'ok'


print(__name__)

# %%
if __name__ == '__main__':
    app.run(debug=True, port=5001) 