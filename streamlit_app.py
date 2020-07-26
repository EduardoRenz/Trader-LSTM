import streamlit as st
from funcoes import *

@st.cache
def load_data(path):
    data = loadNegocios(path) # Carregar os dados de mercado
    data = sugestEntrances(data,spread_minimo=2) # Sugere as entradas
    return data

def build_graph(negocios):
    fig = buildMainGraph(negocios)
    #Ações que devem ser tomadas (comprar e vender)
    for (acao,v) in negocios[negocios.acao != 'do nothing'].groupby('acao'):
        if acao == 'do nothing' :
            continue
        fig.add_trace(go.Scatter(x=v.index,y=v.preco,mode='markers',name=acao, marker={'color':ORDER_COLORS[acao] } ),row=1,col=2)
    
    fig.update_layout(barmode='group',margin=dict(r=10, t=10, b=10, l=10),width=1200)
    return fig


st.title("Day Trade em dolar")



# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data('./dolar/negocios/20200303_dolj20')
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Raw data')
st.dataframe(data.astype('object'))

st.subheader('Gráfico do dia')
st.plotly_chart(build_graph(data))
