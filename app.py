#%%
from flask import Flask,request 
app = Flask(__name__)

@app.route('/negocios',methods=['GET', 'POST'])
def hello_world():
    print(request.data )
    return 'hi'


print(__name__)

# %%
if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000