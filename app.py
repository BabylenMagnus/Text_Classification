from flask import Flask, request, render_template
import pickle
import numpy as np
from text_processing import fix_string


app = Flask(__name__)


class Solution:

    def __init__(self):
        self.tfidf = pickle.load(open('tfidf.pickle', 'rb'))
        self.model = pickle.load(open('logreg.pickle', 'rb'))

    def __call__(self, message: str):
        message = fix_string(message)
        message = self.tfidf.transform(np.array([message]))
        return self.model.predict(message)[0]


sol = Solution()
classes = ["Чат по Python", "Чат по DS"]


@app.route("/", methods=["GET"])
def model_predict():
    args = request.args.to_dict()
    out = sol(list(args.keys())[0])
    return classes[out] + "\n", 201


if __name__ == '__main__':
    app.run(debug=True)
