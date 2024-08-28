import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # Asegúrate de tener esta línea
from sklearn.linear_model import LogisticRegression

# Cargar el archivo Data.json
with open('/Users/miguelangel/Documents/inteligencia artificial/REPOSITORIOS/Chat_AI/Data.json', 'r') as file:
    data = json.load(file)

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar lematizador y vectorizador
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()

# Preprocesar preguntas: tokenización y lematización
preguntas_procesadas = []
respuestas = []
for item in data:
    palabras = nltk.word_tokenize(item['Pregunta'])  # Tokenización usando 'punkt'
    palabras_lemmatizadas = [lemmatizer.lemmatize(w.lower()) for w in palabras]
    preguntas_procesadas.append(' '.join(palabras_lemmatizadas))
    respuestas.append(item['Respuesta'])

# Vectorizar las preguntas
X = vectorizer.fit_transform(preguntas_procesadas)

# Asignar las respuestas como etiquetas (y)
y = respuestas

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Probar el modelo
accuracy = model.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy}')

# Bucle de interacción con el chatbot
while True:
    entrada_usuario = input("Tú: ")
    entrada_procesada = vectorizer.transform([entrada_usuario])
    respuesta_predicha = model.predict(entrada_procesada)[0]
    print(f"Bot: {respuesta_predicha}")
