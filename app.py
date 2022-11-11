import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

df = pd.read_excel("Glucose_Abhishek_ML.xlsx")
x = df.drop(["Glucose_mM"], axis=1)
y = df["Glucose_mM"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
Adaboost_model = AdaBoostRegressor()
# Adaboost_model = LinearRegression()
Adaboost_model.fit(x_train, y_train)
y_pred_Adaboost = Adaboost_model.predict(x_test)
# y_pred_Adaboost = Adaboost_model.predict([[52.1]])
# y_pred_Adaboost

# # using flask_restful
# from flask import Flask, jsonify, request
# from flask_restful import Resource, Api

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Hello(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        return jsonify({'message': 'hello world'})

    # Corresponds to POST request
    def post(self):
        data = request.get_json()  # status code
        return jsonify({'data': data}), 201


# another resource to calculate the square of a number
class Square(Resource):

    def get(self, num):
        return jsonify({'prediction': Adaboost_model.predict(pd.DataFrame([num]))[0]})


# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Square, '/predict/<float:num>')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
