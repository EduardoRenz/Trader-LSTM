
def papel = security(); // Obter o papel atual
def last = papel.last(); // ultimo preço do papel
def bid = papel.bid(); // melhor comprador
def ask = papel.ask(); // melhor vendedor
def bidSize = papel.bidSize(); // qtd de compras no melhor preço de compra
def askSize = papel.askSize(); // qtd de vendas no melhor preço de venda
def change = papel.change(); // variacao percentual
def volume = papel.volume(); // volume negociado


def message = "index;$last;$bid;$bidSize;$ask;$askSize;data";

def post = new URL( "http://127.0.0.1:5000/negocios").openConnection() as HttpURLConnection
post.setRequestMethod("POST")
post.setDoOutput(true)
post.setRequestProperty("Content-Type", "application/text")
post.getOutputStream().write(message.getBytes("UTF-8"));
def postRC = post.getResponseCode();
println(postRC);
if(postRC.equals(200)) {
    println(post.getInputStream().getText());
}




