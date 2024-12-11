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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocessar_texto(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = texto.lower()
    stemmer = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))
    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stop_words]
    return ' '.join(palavras)

modelo_salvo = 'modelo_random_forest.pkl'
vectorizer_salvo = 'vectorizer.pkl'

try:
    model = joblib.load(modelo_salvo)
    vectorizer = joblib.load(vectorizer_salvo)
    logging.info("Modelo e vetorizer carregados com sucesso.")
except FileNotFoundError:
    logging.error("Modelo ou vetorizer não encontrados. Certifique-se de que foram treinados e salvos anteriormente.")

app = Flask(__name__)

@app.route('/recomendar', methods=['POST'])
def recomendar_remedio():
    modelo_salvo2 = 'modelo_random_forest2.pkl'
    vectorizer_salvo2 = 'vectorizer2.pkl'

    try:
        model2 = joblib.load(modelo_salvo2)
        vectorizer2 = joblib.load(vectorizer_salvo2)
        logging.info("Modelo e vetorizer carregados com sucesso.")
    except FileNotFoundError:
        logging.error("Modelo ou vetorizer não encontrados. Certifique-se de que foram treinados e salvos anteriormente.")


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


@app.route('/chat', methods=['POST'])
def recomendar_funcao():
    # Recebendo dados do cliente
    dados = request.get_json()
    mensagem = dados.get('mensagem')

    if not mensagem:
        return jsonify({'error': 'Mensagem não fornecidos'}), 400

    # Pré-processamento
    mensagem_preproc = preprocessar_texto(mensagem)
    mensagem_transf = vectorizer.transform([mensagem_preproc])

    # Fazendo a previsão
    funcao = model.predict(mensagem_transf)[0]
    return jsonify({'funcao': funcao})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
