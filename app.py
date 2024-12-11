from flask import Flask, request, jsonify
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging

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

# Carregando o modelo e o vetorizer salvos
modelo_salvo = 'modelo_random_forest.pkl'
vectorizer_salvo = 'vectorizer.pkl'

try:
    model = joblib.load(modelo_salvo)
    vectorizer = joblib.load(vectorizer_salvo)
    logging.info("Modelo e vetorizer carregados com sucesso.")
except FileNotFoundError:
    logging.error("Modelo ou vetorizer não encontrados. Certifique-se de que foram treinados e salvos anteriormente.")

# Inicializando o Flask
app = Flask(__name__)

# Função para recomendar remédio
@app.route('/recomendar', methods=['POST'])
def recomendar_remedio():
    # Recebendo dados do cliente
    dados = request.get_json()
    sintomas = dados.get('sintomas')

    if not sintomas:
        return jsonify({'error': 'Sintomas não fornecidos'}), 400

    # Pré-processamento
    sintomas_preproc = preprocessar_texto(sintomas)
    sintomas_transf = vectorizer.transform([sintomas_preproc])

    # Fazendo a previsão
    remedio = model.predict(sintomas_transf)[0]
    return jsonify({'remedio': remedio})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
