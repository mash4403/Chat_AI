from flask import Flask, request, jsonify
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Cargar y preprocesar los datos
with open('/Users/miguelangel/Documents/inteligencia artificial/REPOSITORIOS/Chat_AI/Data.json', 'r') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()

preguntas_procesadas = []
respuestas = []
for item in data:
    palabras = nltk.word_tokenize(item['Pregunta'])  # Tokenización usando 'punkt'
    palabras_lemmatizadas = [lemmatizer.lemmatize(w.lower()) for w in palabras]
    preguntas_procesadas.append(' '.join(palabras_lemmatizadas))
    respuestas.append(item['Respuesta'])

X = vectorizer.fit_transform(preguntas_procesadas)
y = respuestas

# Entrenar el modelo
model = LogisticRegression()
model.fit(X, y)

# Ruta para la raíz
@app.route('/', methods=['GET'])
def home():
    return "Chatbot API está funcionando. Usa la ruta /predict para interactuar."

# Ruta para predecir
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['input']
    entrada_procesada = vectorizer.transform([user_input])
    respuesta_predicha = model.predict(entrada_procesada)[0]
    return jsonify({'response': respuesta_predicha})

if __name__ == '__main__':
    app.run(debug=True)
# En este script, hemos creado un servidor web Flask que proporciona una API para interactuar con el chatbot.