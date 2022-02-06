import flask
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
# with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
#     model = pickle.load(f)


app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        print(flask.request.form)
        salary = flask.request.form['salary']
        gender = flask.request.form['gender']
        age = flask.request.form['age']
        role = flask.request.form['role']
        military = flask.request.form['military']
        state = flask.request.form['state']

        comment = flask.request.form.get('comment')

        # input_variables = pd.DataFrame([[salary,gender,age]],
        #                                columns=['salary', 'gender', 'age'],
        #                                # dtype=float
        #                                )
        # prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html')
                                     # original_input={'salary':salary,
                                     #                 'gender':gender,
                                     #                 'age':age},
                                     # result=prediction,
                                     # )


if __name__ == '__main__':
    app.run()
