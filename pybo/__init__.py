from flask import Flask, render_template, redirect, request, url_for
import torch
import numpy as np
import pandas as pd
from tc_learn import *
import joblib
# 여기다가 파이썬 함수 import해오기


def value_to_dict(value):
    input['director'] = value['director']
    input['rating'] = value['rating']
    input['genre'] = value['genre']
    input['company'] = value['company']
    input['writer'] = value['writer']
    input['score'] = value['score']
    input['country'] = value['country']
    return input


def inputChange(input):
    result = {}

    data_company = pd.read_csv('./_csv/company_nm1.csv')
    data_company = data_company.drop(columns='Unnamed: 0')
    data_country = pd.read_csv('./_csv/country_nm1.csv')
    data_country = data_country.drop(columns='Unnamed: 0')
    data_director = pd.read_csv('./_csv/director_nm1.csv')
    data_director = data_director.drop(columns='Unnamed: 0')
    data_genre = pd.read_csv('./_csv/genre_nm.csv')
    data_genre = data_genre.drop(columns='Unnamed: 0')
    data_rating = pd.read_csv('./_csv/rating_nm1.csv')
    data_rating = data_rating.drop(columns='Unnamed: 0')
    data_score = pd.read_csv('./_csv/score_nm1.csv')
    data_score = data_score.drop(columns='Unnamed: 0')
    data_writer = pd.read_csv('./_csv/writer_nm1.csv')
    data_writer = data_writer.drop(columns='Unnamed: 0')

    data_company['0'] = data_company['0'].str.replace(" ", "").str.lower()
    data_country['0'] = data_country['0'].str.replace(" ", "").str.lower()
    data_director['0'] = data_director['0'].str.replace(" ", "").str.lower()
    data_genre['0'] = data_genre['0'].str.replace(" ", "").str.lower()
    data_rating['0'] = data_rating['0'].str.replace(" ", "").str.lower()
    data_writer['0'] = data_writer['0'].str.replace(" ", "").str.lower()

    genre = input.get('genre').strip().lower().replace(' ', '')
    if data_genre[data_genre['0'] == genre].empty:
        result['genre'] = 0
    else:
        genre = data_genre[data_genre['0'] == genre]['zscore']
        result['genre'] = genre.values[0]

    rating = input.get('rating').strip().lower().replace(' ', '')
    if data_rating[data_rating['0'] == rating].empty:
        result['rating'] = 0
    else:
        rating = data_rating[data_rating['0'] == rating]['zscore']
        result['rating'] = rating.values[0]

    country = input.get('country').strip().lower().replace(' ', '')
    if data_country[data_country['0'] == country].empty:
        result['country'] = 0
    else:
        country = data_country[data_country['0'] == country]['zscore']
        result['country'] = country.values[0]

    score = input.get('score').strip().lower().replace(' ', '')
    if data_score[data_score['0'] == float(score)].empty:
        result['score'] = 0
    else:
        score = data_score[data_score['0'] == float(score)]['zscore']
        result['score'] = score.values[0]

    company = input.get('company').strip().lower().replace(' ', '')
    # zscore로 정규화하였기때문에 default값은 0으로...
    if data_company[data_company['0'] == company].empty:
        result['company'] = 0
    else:
        company = data_company[data_company['0'] == company]['zscore']
        result['company'] = company.values[0]

    writer = input.get('writer').strip().lower().replace(' ', '')
    if data_writer[data_writer['0'] == writer].empty:
        result['writer'] = 0
    else:
        writer = data_writer[data_writer['0'] == writer]['zscore']
        result['writer'] = writer.values[0]

    director = input.get('director').strip().lower().replace(' ', '')
    if data_director[data_director['0'] == director].empty:
        result['director'] = 0
    else:
        director = data_director[data_director['0'] == director]['zscore']
        result['director'] = director.values[0]

    return result


def dict_to_arr(result):
    arr = []
    arr.append(result['genre'])
    arr.append(result['rating'])
    arr.append(result['country'])
    arr.append(result['score'])
    arr.append(result['company'])
    arr.append(result['writer'])
    arr.append(result['director'])
    return arr


def DNN(value):
    model = Net()
    #value = value_to_dict(value)
    checkpoint = torch.load('./model/DNN_model.pt')
    model.load_state_dict(checkpoint['model'])
    arr = dict_to_arr(inputChange(value))
    print(arr)
    str = torch.Tensor(arr)
    result = model(str)
    print(result.item())
    return result.item()


def Softmax(value):
    arr = dict_to_arr(inputChange(value))
    print(arr)
    str = torch.Tensor(arr)
    str.unsqueeze_(0)
    W = torch.Tensor([[1.1432, 0.9258, 0.8426, 0.8354, 0.8311],
                     [0.9211, 0.9496, 0.9636, 1.0065, 1.1431],
                      [1.0667, 1.6569, 0.9356, 0.4278, 0.3309],
                      [0.9891, 1.0194, 1.0133, 0.9847, 0.9710],
                      [1.8983, 0.1123, 0.1053, 1.8552, 0.3555],
                      [1.8921, 1.8494, 0.1108, 0.1072, 0.1060],
                      [1.8987, 0.1117, 0.1043, 0.9887, 0.0963]])
    b = torch.Tensor([0.3324])
    result = F.softmax(str.matmul(W) + b, dim=1)
    maxv = 0.0
    maxindex = -1
    for i in range(5):
        if result[0][i].item() > maxv:
            maxindex = i
            maxv = result[0][i].item()
    return maxindex+1


def rfr(input):
    loaded_model = joblib.load('./model/rfr_model_z_t.pkl')

    inputs = pd.DataFrame(inputChange(input), index=[0])

    result = loaded_model.predict(inputs)

    return result[0]


def rfr_h(input):
    loaded_model = joblib.load('./model/rfr_model_z_h.pkl')

    inputs = pd.DataFrame(inputChange(input), index=[0])

    result = loaded_model.predict(inputs)

    return result[0]


def rfc(input):
    loaded_model = joblib.load('./model/rfc_model_z_t1.pkl')

    inputs = pd.DataFrame(inputChange(input), index=[0])

    result = loaded_model.predict(inputs)

    return result[0]


def svm(input):
    loaded_model = joblib.load('./model/svm_model_z_t.pkl')

    inputs = pd.DataFrame(inputChange(input), index=[0])

    result = loaded_model.predict(inputs)

    return result[0]


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index_inyup.html')


@app.route('/resultpage', methods=['POST'])
def post():
    value = request.form

    result_dnn = round(DNN(value), 3)
    result_rfr = round(rfr(value), 3)
    result_rfc = round(rfc(value), 3)
    result_svm = round(svm(value), 3)
    result_softmax = round(Softmax(value), 3)
    
    return render_template('index_wooin.html',result_dnn=result_dnn, result_softmax=result_softmax, tier_dnn=round(result_dnn), tier_softmax=round(result_softmax),
                           result_rfc=result_rfc, result_rfr=result_rfr, result_svm=result_svm,
                            tier_rfc=round(result_rfc), tier_rfr=round(result_rfr), tier_svm=round(result_svm))


if __name__ == '__main__':
    app.run(debug=True)
