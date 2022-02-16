"""

Much of this was sourced from Cambridge Spark's tutorial at https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7

"""

import flask
import pickle
import pandas as pd
from final_model import MLP
# Use pickle to load in the pre-trained model.
# with open(f'f_model.pt', 'rb') as f:
#     model = pickle.load(f)

model = MLP(15)
model.load_state_dict(torch.load('./f_model_v3.pt'))
model.eval()

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
        race = flask.request.form['race']
        state = flask.request.form['state']

        comment = flask.request.form.get('comment')

        input_variables = pd.DataFrame([[state,9999.0,0.0,514.0,age,gender,race,0.0, 0.0, 0.0, military, 1.0, 0.0, 0.0, 9999, 9999, 40, 0.0, 0.0]],
                                       columns=['STATEFIP', 'METAREA', 'OWNERSHP', 'ASECWT', 'AGE', 'SEX', 'RACE', 'MARST', 'POPSTAT',
                                  'ASIAN', 'VETSTAT', 'CITIZEN', 'HISPAN', 'NATIVITY', 'OCC2010', 'CLASSWKR',
                                  'UHRSWORK1', 'PROFCERT', 'EDUC99', 'DIFFANY']
                                       )
        prediction = model.predict(input_variables)[0]
#         prediction = 100
        return flask.render_template('main.html',
                                     original_input={'salary':salary,
                                                     'gender':gender,
                                                     'age':age},
                                     result=prediction-salary,
                                     )


if __name__ == '__main__':
    app.run()
