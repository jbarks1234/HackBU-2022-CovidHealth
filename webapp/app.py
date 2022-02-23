"""

Much of this was sourced from Cambridge Spark's tutorial at https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7

"""
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
import flask
import pickle
import pandas as pd
import torch

from final_model import MLP
# Use pickle to load in the pre-trained model.
# with open(f'f_model.pt', 'rb') as f:
#     model = pickle.load(f)

model = MLP(15)
model.load_state_dict(torch.load('./model/f_model_v3.pt'))
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

        inputs = Tensor([[float(state),float(age),float(gender),float(race),float(emphlth),float(milhlth),float(military),float(nativ), 0.0, float(job), float(emp), float(hours), float(prof), float(educ), float(privhlth)]])
        # prediction = model.predict(input_variables)[0]

        # make prediction
        yhat = model(inputs)
        # retrieve numpy array
        scaler = MinMaxScaler()
        yhat = yhat.detach().numpy()
        # yhat = (yhat-2.0)/(2099999.0-2.0)
        yhat = (yhat[0][0]*(1999999.0-2.0))+2.0
        prediction = str(int(yhat))

        prediction = "{:,}".format(int(prediction))
        # normalized_df=(df-df.min())/(df.max()-df.min())


        return flask.render_template('main.html',
                                     original_input={'Your salary':salary},
                                     result='$ '+prediction,
                                     )


if __name__ == '__main__':
    app.run()
