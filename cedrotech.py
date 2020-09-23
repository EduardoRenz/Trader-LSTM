
import asyncio
import websockets
import json 
token = ''
uri = "ws://webfeeder.cedrotech.com/ws"
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
        "parameters": {"subsbribetype": "1", "filter": "2,3,4", "delay":"400"}
    }  

    await websocket.send(json.dumps(data))
    response = await websocket.recv()
    return response

async def getTrades(websocket,token,active_code):
    data = {"token": token,  
        "module": "quotes",  
        "service": "quoteTrade",  
        "parameterGet": active_code,  
        "parameters": {"subsbribetype": "1", "quantidade": "2" ,"delay": "600"}
    }  

async def getBook(websocket,token,active_code):
    data = {"token": token,  
        "module": "quotes",  
        "service": "aggregatedBook",  
        "parameterGet": active_code,  
        "parameters": {"subsbribetype": "1", "delay":"5000"}
    }  




async def consume_data(websocket, path):
    async for message in websocket:
        print(message)

# Faz conexão com o servidor
async def connect_to_server():
    global token
    async with websockets.connect(uri) as websocket:
        token = await login(websocket)
        quote = await getQuote(websocket,token,"petr4")
        trades = await getTrades(websocket,token,"petr4")
        book = await getBook(websocket,token,"petr4")
        print(token)
        print(quote)
        print(trades)
        print(book)
        
        #await consume_data(websocket,'')
        async for message in websocket:
            print(message)






asyncio.get_event_loop().run_until_complete(connect_to_server())
#asyncio.get_event_loop().run_until_complete(get_quote())
#asyncio.get_event_loop().run_forever()