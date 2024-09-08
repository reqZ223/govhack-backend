from flask import Flask, request, jsonify
from govHack import provide_output
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/submit', methods=['POST'])
def rewrite_data():
    # Get JSON data from the request
    data = request.get_json()
    question = data.get('question')
    type = data.get('type')
    department = "home_affairs" # Hardcoded for now
    #department = data.get('department')
    answer = provide_output(question, type, department)
    return jsonify({"answer": answer}), 200


if __name__ == '__main__':
    app.run(debug=True)