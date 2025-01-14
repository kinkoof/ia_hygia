import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging
from datetime import datetime, timedelta

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# URL da API CoinGecko para obter o preço atual do BNB
API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd"

# Nome do arquivo CSV
ARQUIVO_CSV = 'bnb_prices.csv'

# Nome do modelo salvo
MODELO_SALVO = 'modelo_previsao_preco.pkl'

# Função para processar a coluna Date e extrair recursos temporais
def processar_datas(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')  # Converter para formato datetime
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data['weekday'] = data['Date'].dt.weekday  # Dia da semana (0=segunda-feira)
    return data

# Função para obter o preço atual da API
def obter_preco_atual():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        preco_atual = response.json()['binancecoin']['usd']
        logging.info(f"Preço atual do BNB obtido: {preco_atual:.2f} USD")
        return preco_atual
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao obter o preço atual: {e}")
        return None

# Função para atualizar o CSV com o preço do dia atual
def atualizar_csv(preco_atual):
    hoje = datetime.now().strftime('%d/%m/%Y')
    try:
        # Carregar o CSV
        data = pd.read_csv(ARQUIVO_CSV)
        if hoje in data['Date'].values:
            logging.info(f"Preço para {hoje} já registrado no CSV.")
        else:
            # Adicionar o preço atual ao CSV
            novo_registro = pd.DataFrame({'Date': [hoje], 'Price': [preco_atual]})
            data = pd.concat([data, novo_registro], ignore_index=True)
            data.to_csv(ARQUIVO_CSV, index=False)
            logging.info(f"Preço para {hoje} adicionado ao CSV com sucesso.")
    except FileNotFoundError:
        # Criar o CSV se não existir
        logging.warning(f"{ARQUIVO_CSV} não encontrado. Criando um novo arquivo.")
        novo_registro = pd.DataFrame({'Date': [hoje], 'Price': [preco_atual]})
        novo_registro.to_csv(ARQUIVO_CSV, index=False)
        logging.info(f"{ARQUIVO_CSV} criado e preço registrado.")

# Função para treinar ou carregar o modelo
def carregar_ou_treinar_modelo():
    try:
        # Carregar o modelo salvo
        model = joblib.load(MODELO_SALVO)
        logging.info("Modelo carregado com sucesso.")
    except FileNotFoundError:
        # Carregar os dados do CSV
        data = pd.read_csv(ARQUIVO_CSV)
        data = processar_datas(data)

        # Separar recursos (X) e alvo (y)
        X = data[['year', 'month', 'day', 'weekday']]  # Recursos derivados da data
        y = data['Price']  # Alvo é o preço

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo
        logging.info("Modelo não encontrado, treinando um novo modelo...")
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Avaliar o modelo
        y_pred = model.predict(X_test)
        erro_medio = mean_absolute_error(y_test, y_pred)
        logging.info(f"Erro médio absoluto no conjunto de teste: {erro_medio:.2f}")

        # Salvar o modelo treinado
        joblib.dump(model, MODELO_SALVO)
        logging.info("Modelo salvo com sucesso.")
    return model

# Obter o preço atual do BNB
preco_atual = obter_preco_atual()
if preco_atual is None:
    logging.error("Não foi possível obter o preço atual. Finalizando o programa.")
    exit()

# Atualizar o CSV com o preço atual
atualizar_csv(preco_atual)

# Carregar ou treinar o modelo
modelo = carregar_ou_treinar_modelo()

# Prever o preço do próximo dia
amanha = (datetime.now() + timedelta(days=1)).strftime('%d/%m/%Y')
hoje_processado = pd.DataFrame([{
    'year': datetime.now().year,
    'month': datetime.now().month,
    'day': datetime.now().day,
    'weekday': datetime.now().weekday()
}])
preco_previsto = modelo.predict(hoje_processado)[0]
logging.info(f"Preço previsto para {amanha}: {preco_previsto:.2f} USD")
print(f"Preço previsto para {amanha}: {preco_previsto:.2f} USD")