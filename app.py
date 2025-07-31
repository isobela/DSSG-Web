from flask import Flask, request, jsonify, render_template
from rag.query_data_pc import query_rag  # Your RAG logic

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')  # or home.html if you have one

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/model')
def model():
    return render_template('model.html')  # This loads model.html from /templates

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('query')
    answer = query_rag(user_query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

