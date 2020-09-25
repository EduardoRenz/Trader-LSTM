
import asyncio
import websockets
import json 
from datetime import datetime
token = ''
uri = "ws://webfeeder.cedrotech.com/ws"
ativo = 'dolv20'
## =============================== FUNCTIONS ===============================
async def login(websocket):
    await websocket.send("""{  
                    "module": "login",  
                    "service": "authentication",  
                    "parameters": {"login": "dudarenz", "password": "102030"}  
                }""")
    login = await websocket.recv()
    data =  json.loads(login)
    return data['token']


async def getQuote(websocket,token,active_code):
    data = {"token": token,  
        "module": "quotes",  
        "service": "quote",  
        "parameterGet": active_code,  
        "parameters": {"subsbribetype": "1",  "delay":"400"}
    }  

    await websocket.send(json.dumps(data))
    response = await websocket.recv()
    return json.loads(response)

async def getTrades(websocket,token,active_code):
    data = {"token": token,  
        "module": "quotes",  
        "service": "quoteTrade",  
        "parameterGet": active_code,  
        "parameters": {"subsbribetype": "1", "quantidade": "1"}
    }  
    await websocket.send(json.dumps(data))
    response = await websocket.recv()
    return json.loads(response)

async def getBook(websocket,token,active_code):
    data = {"token": token,  
        "module": "quotes",  
        "service": "aggregatedBook",  
        "parameterGet": active_code,  
        "parameters": {"subsbribetype": "1", "delay":"5000"}
    }  
    await websocket.send(json.dumps(data))
    response = await websocket.recv()
    return json.loads(response)


async def saveTrades(data):
    if(("quoteTrade" not in data) or  (len(data["quoteTrade"]["L"]) == 0)):
        return
    time = datetime.strptime(data["quoteTrade"]["L"][0]['T'],'%b %d, %Y %I:%M:%S %p')
    with open(f'inputs/{data["quoteTrade"]["M"]}_{time.strftime("%Y_%m_%d")}_trades.json', 'a+') as outfile:
        trades = data["quoteTrade"]["L"]
        for trade in trades:
            outfile.write(json.dumps(trade)+",")
        return 'ok'

async def saveBook(data):
    if('book' not in data):
        return
    #time = datetime.strptime(data["L"][0]['T'],'%b %d, %Y %I:%M:%S %p')
    with open(f'inputs/{data["book"]["S"]}_book.json', 'a+') as outfile:
        books = data["book"]["B"]
        for book in books:
            book["T"] = datetime.now()
            outfile.write(json.dumps(book, default=str)+",")
        return 'ok'

async def saveQuotes(data):
    if(("values" not in data) ):
        return
    with open(f'inputs/{data["parameter"]}_quotes.json', 'a+') as outfile:
        data["values"]["T"] = datetime.now()
        outfile.write(json.dumps(data,default=str)+",")
        return 'ok'

# Faz conexão com o servidor
async def connect_to_server():
    global token
    async with websockets.connect(uri) as websocket:
        token = await login(websocket)
        quote = await getQuote(websocket,token,ativo)
        trades = await getTrades(websocket,token,ativo)
        await saveTrades(trades)
        await getBook(websocket,token,ativo)
        #print(token)
        #print(quote)
        #print(trades)
        #print(book)
        
        #await consume_data(websocket,'')
        async for message in websocket:
            data = json.loads(message)
            await saveBook(data)
            await saveTrades(data)
            await saveQuotes(data)
            #saveTrades(message)
            #print(data)






asyncio.get_event_loop().run_until_complete(connect_to_server())
#asyncio.get_event_loop().run_until_complete(get_quote())
#asyncio.get_event_loop().run_forever()