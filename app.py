from flask import Flask,render_template,request,url_for,redirect,abort
import flasgger
from werkzeug.utils import secure_filename
from flasgger import Swagger
import _pickle as cPickle
import pandas as pd
import scipy as sc
import csv
import os
import re

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectMultipleField
from wtforms.validators import DataRequired
from collections import Counter
from preprocessing import Clustering
import matplotlib
matplotlib.use('Agg')

app=Flask(__name__) 
app.config['SECRET_KEY'] = '3674urgakjt7s'
# Bootstrap(app)

#First Page

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file_1=request.files['csvfile1']
        file_2=request.files['csvfile2']
        file_1.save('Provide location for the saving the file 1  in static folder')
        file_2.save('Provide location for the saving the file 2  in static folder')
        return redirect(url_for('feature_link'))


    return render_template('index.html')


@app.route("/feature_link",methods=['GET','POST']) # decorator
def feature_link():

    user_file=pd.read_csv('Load the file 1')
    user_columns_list=user_file.columns.tolist()

    access_file=pd.read_csv('Load the file 2')
    access_columns_list=access_file.columns.tolist()

     
    return render_template('feature_link.html',data1=user_columns_list,data2=access_columns_list)

#Second Page 

@app.route("/columns",methods=['GET','POST']) # decorator
def columns_name_recieved():
    user_form_columns=[]
    if request.method =='POST':
        user_form_columns=request.form.getlist("usercolumns")
        access_form_columns=request.form.getlist("accesscolumns")

        for i in range(len(access_form_columns)):
            access_form_columns[i] = access_form_columns[i][1:]

        obj = Clustering('provide the file 2 address ',access_form_columns,'Provide the File 1 address',user_form_columns)
        output = obj.preprocess()
        

    return render_template('columns.html',data3= output, ent_cols = access_form_columns, user_cols = user_form_columns)


#Third page and cluster link page
@app.route("/details/<data1>/<data2>")
def cluster_details(data1,data2):
    
    user_file=pd.read_csv('Read the File 1')
    col = user_file.columns.tolist()
    df = user_file[user_file[col[0]].isin(data2[1:].replace("""'""","").split(" "))].values
    

    #to take digit from the cluster
    digit=re.search(r'\d+', data1).group()
    number=digit


    images='Mention address of images stored in the static folder'

    user_images_regex=re.compile('^user_' + (number))
    eval_images_regex=re.compile('^eval_' + (number))


    rootdir = 'Address of the static folder'

    user_images=[]
    eval_images=[]
    for  root, dirs,files in os.walk(rootdir):
        for file in files:
            if user_images_regex.match(file):
                user_images.append(file)
            elif eval_images_regex.match(file):
                eval_images.append(file)
                

    return render_template("details.html", data5 = data1, data6=df, data7=col,data8=user_images,data9=eval_images)


#outlier page

@app.route("/outlier/<data>",methods=['GET','POST']) 
def outlier(data):
    if request.method=='POST':
        results=pd.read_csv('Load the results.csv file ')
        id_column = results.columns[0]
        change_df = results[results['update']=='change'].groupby('labels')[id_column].count().reset_index(name='count')
        change_df["List of IDs"]=results[results['update']=='change'].groupby('labels')[id_column].apply(list).values


    return render_template("outlier.html",column_names=change_df.columns.values, row_data=list(change_df.values.tolist()),
                           link_column="List of IDs",zip=zip , data_x = data)



#outlier users page
@app.route("/outlier/<user_id>/<ent_cols>")
def user_details(user_id,ent_cols):
    ent_cols = ent_cols[1:-1].replace("""'""","").split(",")

    results=pd.read_csv('Load the result_data.csv file saved in static folder')
    
    temp_dict={ }
    cols = results.columns.tolist()
    for col,val in zip(cols,results[results[cols[0]]==user_id].values[0]):
        if col in ['ent_remain','ent_add','ent_remove']:
            new_val = val[1:-1].replace("""'""","").split(",")
            for i in range(1,len(new_val)):
                new_val[i]=new_val[i].split(" / ")
            val = new_val
        temp_dict[col] = val

    return render_template("user.html",user_id = user_id, user_data = temp_dict, ent_cols = ent_cols)


#recommendation Page
@app.route("/predictions/<ent_cols>/<user_cols>",methods=['GET','POST']) 
def predictions(ent_cols,user_cols):
    user_cols = user_cols[1:-1].replace("""'""","").split(",")
    
    user_data=pd.read_csv('load the file 1')

    select_dict = {}
    for i in user_cols[1:]:
        x = user_data[i[1:]].unique().tolist()
        for k in range(len(x)):
            x[k] = x[k].replace(' ','_')
        select_dict[i[1:]] = x
    

    return render_template("predictions.html",select_user=select_dict,ent_cols=ent_cols)

@app.route('/pred_results/<data>/<ent_cols>', methods=['GET', 'POST'])
def pred_results(data,ent_cols):
    col_names = ent_cols[1:-1].replace("""'""","").split(",")
    data = data[11:-2].replace("""'""","").split(",")

    dummy=[]
    for i in data:
        if i[0]==' ':
            i = i[1:]
        dummy.append(i+'_'+request.form.get(i, None))

    user_data=pd.read_csv('Load the Clust_vector.csv file')
    
    user_vector = user_data[user_data.index ==0]['user_vector'].values[0][:-1].replace("""'""","").split(",")
    new_user_vector = []

    for i in user_vector:
        if i[1:].replace(' ','_') in dummy:
            new_user_vector.append(1)
        else:
            new_user_vector.append(0)

    max_sim = 0 
    for i in user_data[user_data.index >0].values:
        user_vect = i[2][1:-1].split(",")
        user_vect = [int(x) for x in user_vect]
        sim = 1- sc.spatial.distance.cosine(new_user_vector,user_vect)

        if sim >= max_sim:
            max_sim = sim
            cluster = i[0]
            new_entit = i[1][1:-1].split(",")
    
    new_entit = [int(x) for x in new_entit]
    new_entitlement = []
    ent_vect = user_data[user_data.index ==0]['ent_vector'].values[0][1:-1].replace("""'""","").split(",")
    len(ent_vect)
    for i,j in zip(ent_vect,new_entit):
        if j==1:
            new_entitlement.append(i.split("/"))
    


    
    return render_template("pred_results.html",inp_data=new_entitlement,names=col_names[1:])


if __name__ == "__main__":
    app.run(debug=True)
    



