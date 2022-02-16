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
        job = flask.request.form['job']
        military = flask.request.form['military']
        emp = flask.request.form['emp']
        emphlth = flask.request.form['emphlth']
        privhlth = flask.request.form['privhlth']
        milhlth = flask.request.form['milhlth']
        race = flask.request.form['race']
        nativ = flask.request.form['nativ']
        hours = flask.request.form['hours']
        educ = flask.request.form['educ']
        prof = flask.request.form['prof']
        state = flask.request.form['state']

        input_variables = pd.DataFrame([[state,age,gender,race,emphlth,milhlth,military,nativ, 0.0, job, emp, hours, prof, educ, privhlth]],
                                       columns=['STATEFIP', 'AGE', 'SEX', 'RACE', 'PAIDGH', 'HICHAMP',
                                                'VETSTAT', 'NATIVITY', 'HISPAN', 'OCC2010', 'EMPSAME',
                                                'UHRSWORK1', 'PROFCERT', 'EDUC99', 'PHINSUR'])
        prediction = model.predict(input_variables)[0]
#         prediction = 100
        return flask.render_template('main.html',
                                     original_input={'salary':salary,
                                                     'gender':gender,
                                                     'age':age,
                                                     'job':job,
                                                     'military':military,
                                                     'emp':emp,
                                                     'emphlth':emphlth,
                                                     'privhlth':privhlth,
                                                     'milhlth':milhlth,
                                                     'race':race,
                                                     'nativ':nativ,
                                                     'hours':hours,
                                                     'educ':educ,
                                                     'prof':prof,
                                                     'state':state},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
