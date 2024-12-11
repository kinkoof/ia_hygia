import re
import pandas as pd
import logging
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import nltk
from collections import Counter

nltk.download('rslp')
nltk.download('stopwords')

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função de pré-processamento
def preprocessar_texto(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = texto.lower()
    stemmer = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))
    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stop_words]
    return ' '.join(palavras)

# Carregando os dados
data = pd.read_csv('remedios.csv')

# Aplicando pré-processamento nos sintomas
data['Descrição dos Sintomas'] = data['Descrição dos Sintomas'].apply(preprocessar_texto)

# Vetorização de sintomas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Descrição dos Sintomas'])
y = data['Médico Recomendado']

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibindo a distribuição inicial das classes
classe_contagem = Counter(y_train)
logging.info(f"Distribuição inicial das classes: {classe_contagem}")

# Verificando se o modelo já existe
modelo_salvo = 'modelo_random_forest.pkl'
vectorizer_salvo = 'vectorizer.pkl'

try:
    # Tentando carregar o modelo e o vetorizer salvos
    model = joblib.load(modelo_salvo)
    vectorizer = joblib.load(vectorizer_salvo)
    logging.info("Modelo e vetorizer carregados com sucesso.")
except FileNotFoundError:
    # Treinando o modelo caso não encontre os arquivos
    logging.info("Modelo não encontrado, treinando o modelo...")
    # Treinando o modelo Random Forest
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # Salvando o modelo e o vetorizer
    joblib.dump(model, modelo_salvo)
    joblib.dump(vectorizer, vectorizer_salvo)
    logging.info("Modelo e vetorizer salvos com sucesso.")

# Fazendo previsões
y_pred = model.predict(X_test)

# Função para recomendar remédio
def recomendar_remedio(sintomas):
    logging.info(f"Sintomas recebidos: {sintomas}")
    sintomas_preproc = preprocessar_texto(sintomas)
    sintomas_transf = vectorizer.transform([sintomas_preproc])
    remedio = model.predict(sintomas_transf)[0]
    logging.info(f"Remédio recomendado: {remedio}")
    return remedio

# Exemplo de uso
sintomas_usuario = "olhos inchados"
remedio_recomendado = recomendar_remedio(sintomas_usuario)
print(f"Remédio recomendado: {remedio_recomendado}")
