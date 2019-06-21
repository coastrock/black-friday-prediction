from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_purchase', methods=['POST', 'GET'])
def predict_purchase():
    # Getting the parameters Gender	Age Occupation City_Category Stay_In_Current_City_Years Marital_Status Product_Category_1
    customer_gender = str(request.form['customer_gender'])
    customer_age = str(request.form['customer_age'])
    customer_occupation = int(request.form['customer_occupation'])
    city_category = str(request.form['city_category'])
    time_lived_in_city = str(request.form['time_lived_in_city'])
    marital_status = int(request.form['marital_status'])
    product_category_1 = int(request.form['product_category_1'])

    # Loading the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # Generating a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)
    
    # Changing the dataframe according to the inputs
    df_prediction.at[0, 'Gender_'+str(customer_gender)] = 1.0
    df_prediction.at[0, 'Age_'+str(customer_age)] = 1.0
    df_prediction.at[0, 'Occupation_'+str(customer_occupation)] = 1.0
    df_prediction.at[0, 'City_Category_'+str(city_category)] = 1.0
    df_prediction.at[0, 'Stay_In_Current_City_Years_'+str(time_lived_in_city)] = 1.0
    df_prediction.at[0, 'Marital_Status_'+str(marital_status)] = 1.0
    df_prediction.at[0, 'Product_Category_1_'+str(product_category_1)] = 1.0
    print(df_prediction)

    # Loading the model and predicting
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_purchase = prediction.round(1)[0]

    if customer_gender == 'F':
      customer_gender = 'female'
    elif customer_gender == 'M':
      customer_gender = 'male'

    if customer_age == '0-17':
      customer_age = 'between 0 and 17 years old'
    elif customer_age == '18-25':
      customer_age = 'between 18 and 25 years old'
    elif customer_age == '26-35':
      customer_age = 'between 26 and 35 years old'
    elif customer_age == '36-45':
      customer_age = 'between 36 and 45 years old'
    elif customer_age == '46-50':
      customer_age = 'between 46 and 50 years old'
    elif customer_age == '51-55':
      customer_age = 'between 51 and 55 years old'
    elif customer_age == '55+':
      customer_age = 'older than 55 years old'
    
    if customer_occupation == 0:
      customer_occupation = 'accountants'
    elif customer_occupation == 1:
      customer_occupation = 'business entrepreneurs'
    elif customer_occupation == 2:
      customer_occupation = 'coaches'
    elif customer_occupation == 3:
      customer_occupation = 'dentists'
    elif customer_occupation == 4:
      customer_occupation = 'engineers'
    elif customer_occupation == 5:
      customer_occupation = 'financial analysts'
    elif customer_occupation == 6:
      customer_occupation = 'graphic designers'
    elif customer_occupation == 7:
      customer_occupation = 'historians'
    elif customer_occupation == 8:
      customer_occupation = 'judges'
    elif customer_occupation == 9:
      customer_occupation = 'lawyers'
    elif customer_occupation == 10:
      customer_occupation = 'interns'
    elif customer_occupation == 11:
      customer_occupation = 'musicians'
    elif customer_occupation == 12:
      customer_occupation = 'nurses'
    elif customer_occupation == 13:
      customer_occupation = 'ophthalmologists'
    elif customer_occupation == 14:
      customer_occupation = 'physicists'
    elif customer_occupation == 15:
      customer_occupation = 'reporters'
    elif customer_occupation == 16:
      customer_occupation = 'singers'
    elif customer_occupation == 17:
      customer_occupation = 'surgeons'
    elif customer_occupation == 18:
      customer_occupation = 'teachers'
    elif customer_occupation == 19:
      customer_occupation = 'technical managers'
    elif customer_occupation == 20:
      customer_occupation = 'web developers'
    
    if city_category == 'A':
      city_category = 'Atlanta'
    elif city_category == 'B':
      city_category = 'Boston'
    elif city_category == 'C':
      city_category = 'Chicago'
    
    if time_lived_in_city == '0':
      time_lived_in_city = 'less than 1 year'
    elif time_lived_in_city == '1':
      time_lived_in_city = '1 year'
    elif time_lived_in_city == '2':
      time_lived_in_city = '2 years'
    elif time_lived_in_city == '3':
      time_lived_in_city = '3 years'
    elif time_lived_in_city == '4+':
      time_lived_in_city = '4 years or more'
    
    if marital_status == 0:
      marital_status = 'Single'
    elif marital_status == 1:
      marital_status = 'Married'
    
    if product_category_1 == 1:
      product_category_1 = 'TVs'
    elif product_category_1 == 2:
      product_category_1 = 'Laptops & Computers'
    elif product_category_1 == 3:
      product_category_1 = 'Tablets & E-Readers'
    elif product_category_1 == 4:
      product_category_1 = 'Video Games, Consoles & VR products'
    elif product_category_1 == 5:
      product_category_1 = 'Headphones, Earbuds & Speakers'
    elif product_category_1 == 6:
      product_category_1 = 'Cell Phones'
    elif product_category_1 == 7:
      product_category_1 = 'Home Theater, Audio & Video products'
    elif product_category_1 == 8:
      product_category_1 = 'Smart Home, Security & Wi-Fi products'
    elif product_category_1 == 9:
      product_category_1 = 'Wearable Technology products'
    elif product_category_1 == 10:
      product_category_1 = 'Drones'
    elif product_category_1 == 11:
      product_category_1 = 'Cameras & Camcorders'
    elif product_category_1 == 12:
      product_category_1 = 'Activity Trackers & Smartwatches'
    elif product_category_1 == 13:
      product_category_1 = 'Major Appliances'
    elif product_category_1 == 14:
      product_category_1 = 'Small Appliances'
    elif product_category_1 == 15:
      product_category_1 = 'Movies, TV Shows & Music products'
    elif product_category_1 == 16:
      product_category_1 = 'Printers, Ink & Home Office products'
    elif product_category_1 == 17:
      product_category_1 = 'Car Electronics & GPS products'
    elif product_category_1 == 18:
      product_category_1 = 'Health & Fitness Gadgets'    
    
    return render_template('results.html',
                           customer_gender=str(customer_gender),
                           customer_age=str(customer_age),
                           customer_occupation=str(customer_occupation),
                           city_category=str(city_category),
                           time_lived_in_city=str(time_lived_in_city),
                           marital_status=str(marital_status),   
                           product_category_1=str(product_category_1),
                           predicted_purchase="{:,}".format(predicted_purchase)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
