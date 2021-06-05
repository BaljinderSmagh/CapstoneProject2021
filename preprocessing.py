import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import scipy as sc
import matplotlib as mpl
import itertools
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, FastICA, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Import scikitlearn for machine learning functionalities

from sklearn.manifold import TSNE 

import seaborn as sns; sns.set(style='white')
from sklearn.metrics import davies_bouldin_score
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import yellowbrick
from yellowbrick.cluster import KElbowVisualizer

import psutil
import plotly.express as px
import os
import re


class Clustering:

    #getting the inputs
    def __init__(self, access_file, target_columns, user_file, varified_columns):
        self.access_file = access_file
        self.target_col = target_columns
        self.user_file = user_file
        self.varified_cols = varified_columns
        self.vector_mat = []

    def data_read(self,data_file,columns):
        dummy_df = pd.read_csv(data_file, encoding='cp1252')
        return dummy_df[columns]
        
    #one hot encoding
    def feature_engineering(self):
        self.access_data = self.access_data.join(pd.get_dummies(self.access_data['ent_combo']))
        self.access_data = self.access_data.drop('ent_combo',axis = 1)
    
        self.access_data = self.access_data.groupby(self.target_col[0]).sum().reset_index()
        self.clust_data = self.access_data.iloc[:,1:]

    #Kmeans 
    def n_clusters(self):
        inertia = []
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=1).fit(self.clust_data)
            inertia.append(kmeans.inertia_)
        
        plt.plot(range(1, 20), inertia, marker='s');
        plt.xlabel('$k$')
        plt.ylabel('$J(C_k)$')
        plt.savefig('KMeans_clusters.png', dpi=100, bbox_inches='tight')

    #3d cluster visualization
    def cluster_results(self):
        # KMeans Clustring
        number_of_cluster = 6
        kmeans_10 = KMeans(n_clusters=number_of_cluster,random_state=0)
        kmeans_10.fit(self.clust_data)
        self.result_labels = kmeans_10.predict(self.clust_data)

        # Let's create a beautiful 3d-plot
        fig = plt.figure(1, figsize=(10, 10))
        plt.clf()
        ax = Axes3D(fig, rect=[1, 1, 1, 1])
        plt.cla()

        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(self.clust_data)
        PCA_components = pd.DataFrame(principalComponents)
        # Change the order of labels, so that they match
        ax.scatter(PCA_components[0], PCA_components[1], PCA_components[2],c=self.result_labels ,marker = '^',alpha=0.6,
                cmap='jet_r')

        ax.view_init(10,65)
        ax.set_xlabel('$PCA_1$')
        ax.set_ylabel('$PCA_2$')
        ax.set_zlabel('$PCA_3$')

        ax.set_title('3D Visualization of users in Hyper Space',fontsize=20, fontweight ='bold')
        
        plt.savefig('static path folder for png/cluster_Results.png', dpi=100, bbox_inches='tight')
    

    #Outlier detection algorithm
    def finding_outliers(self,number_of_cluster):
        outlierUID={}
        for cluster in range(number_of_cluster):
            matrix_1 = self.user_info[self.user_info['labels']==cluster]
            mat =[]
            for i in matrix_1[self.varified_cols[0]].values:
                row = [i]
                for j in matrix_1[self.varified_cols[0]]:
                    row.append(1- sc.spatial.distance.cosine(matrix_1[matrix_1[self.varified_cols[0]]==i]['info_vector'].values[0], matrix_1[matrix_1[self.varified_cols[0]]==j]['info_vector'].values[0] ))
                mat.append(row)
            mat_df = pd.DataFrame(mat) 
            mat_df.columns = ['index']+ list(matrix_1[self.varified_cols[0]].values)
            mat_df = mat_df.set_index('index')
            mat_df['sum']=mat_df.sum(axis=1)
            mat_df['sum']=mat_df['sum'].apply(lambda x:x-1)
            mat_df[self.varified_cols[0]]=mat_df.index
            q75, q25 = np.percentile(mat_df['sum'].values, [75 ,25])
            iqr = q75 - q25
            outlierUID[cluster] = mat_df[(mat_df['sum']<(q25-1.5*iqr)) | (mat_df['sum']>(q75+1.5*iqr))][self.varified_cols[0]].values
            dummy = []
            for val in self.user_info[self.user_info['labels']==cluster][self.varified_cols[0]].values:
                if val in outlierUID[cluster]:
                    dummy.append(1)
                else:
                    dummy.append(0)
        uid_outliers=[]
        for i,j in outlierUID.items():
            for _ in j:
                uid_outliers.append(_)
        
        return outlierUID, uid_outliers
    
    #recommending the correct group for the outlier users
    def relocating_outliers(self,number_of_cluster):
        self.info_score_dict={}
        user_alpha = 0.05
        for i in range(number_of_cluster):
            user = list(self.user_info_wo_outlier[self.user_info_wo_outlier['labels']==i]['info_vector'].values)
            user_info_score = list(map(sum,zip(*user)))
            for j in range(len(user_info_score)):
                if user_info_score[j] >= user_alpha*len(self.user_info_wo_outlier[self.user_info_wo_outlier['labels']==i]):
                    user_info_score[j]=1
                else:
                    user_info_score[j]=0
            self.info_score_dict[i]=user_info_score
            
        new_cluster = [] 
        for score in self.outlier.info_vector.values:
            clust = 0
            max_sim = 0
            for i in range(number_of_cluster):
                sim = 1-sc.spatial.distance.cosine(score,self.info_score_dict[i])
                if sim>max_sim:
                    clust = i
                    max_sim=sim
            new_cluster.append(clust)
        
        
        
        self.outlier['new_labels'] = new_cluster
        self.outlier['update'] = np.where(self.outlier['labels']==self.outlier['new_labels'],'not change','change')
        
    #recommending entitlement change for outlier users
    def entitlement_recommendation(self, number_of_cluster):
        self.entitlement_list = self.access_data.columns[1:-1]
        self.ent_score_dict={}
        ent_alpha = 0.1
        for i in range(number_of_cluster):
            ent_info = self.access_info_wo_outlier[self.access_info_wo_outlier['labels']==i].values[:,1:-1]
            ent_info_score = ent_info.sum(axis=0).tolist()

            for j in range(len(ent_info_score)):
                if ent_info_score[j] >= ent_alpha*len(self.access_info_wo_outlier[self.access_info_wo_outlier['labels']==i]):
                    ent_info_score[j]=1
                else:
                    ent_info_score[j]=0
            self.ent_score_dict[i]=ent_info_score
        
        existing =[]
        remove = []
        add =[]
        for uid in self.outlier[self.varified_cols[0]].values:
            e=[]
            r=[]
            a=[]
            user_ent = self.access_outlier[self.access_outlier[self.varified_cols[0]]==uid].values[0][1:-1]
            new_cluster = self.outlier[self.outlier[self.varified_cols[0]]==uid]['new_labels'].values[0]
            update = self.outlier[self.outlier[self.varified_cols[0]]==uid]['update'].values[0]
            for i in range(len(user_ent)):
                if update =='not change'and user_ent[i]>0:
                    e.append(self.entitlement_list[i])
                elif update =='not change'and user_ent[i]==0:
                    pass
                elif user_ent[i]==self.ent_score_dict[new_cluster][i] and user_ent[i]>0:
                    e.append(self.entitlement_list[i])
                elif user_ent[i]>self.ent_score_dict[new_cluster][i]:
                    r.append(self.entitlement_list[i])
                elif user_ent[i]<self.ent_score_dict[new_cluster][i]:
                    a.append(self.entitlement_list[i])
            e.insert(0,len(e))
            r.insert(0,len(r))
            a.insert(0,len(a))
            existing.append(e)
            remove.append(r)
            add.append(a)

        self.outlier['ent_remain']=existing
        self.outlier['ent_remove']=remove
        self.outlier['ent_add']=add
    
    #cluserted data results
    def vector_data(self, number_of_cluster):
        info_mat=[[np.nan,self.entitlement_list.tolist(), self.job_id.tolist()]]
        for i in range(number_of_cluster):
            row_1=[i]
            row_1.append(self.ent_score_dict[i])
            row_1.append(self.info_score_dict[i])
            info_mat.append(row_1)
            
        dum_df = pd.DataFrame(info_mat, columns = ['Cluster','ent_vector','user_vector'])
        dum_df.to_csv('static path folder for /Clust_vector.csv file',index=False)
    

    #visualization of clusters distribution for user and access file attributes
    def charts(self,access_file,user_file,number_of_clusters):
        user_images_regex=re.compile('^user_')
        eval_images_regex=re.compile('^eval_')
        for fname in os.listdir('Mention path for static folder /static/'):
            file_path = os.path.abspath(os.path.join('/Mention path for static folder /static/', fname))
            if user_images_regex.match(fname) or eval_images_regex.match(fname):
                os.remove(file_path)
        for j in range(number_of_clusters):
            data1=access_file[access_file['labels']==j]
            data2=user_file[user_file['labels']==j]
        
            for i, val in enumerate(self.target_col[1:]):
                temp1 = data1[val].value_counts()
                temp2 = temp1.head(5)
                temp = temp2.rename_axis(val).reset_index(name='counts')

                fig = px.pie(temp, names=val, values="counts", title=val)
                fig.write_image("Mention path for static folder /static/eval_{}{}.png".format(j,i))

            
            for i, val in enumerate(self.varified_cols[1:]):
                temp1 = data2[val].value_counts()
                temp2 = temp1.head(5)
                if len(temp1) > 5:
                        temp2['Remaining {0} items'.format(len(temp1) - 5)] = sum(temp1[5:])
                temp = temp2.rename_axis(val).reset_index(name='counts')
                fig = px.pie(temp, names=val, values="counts", title=val)
                fig.write_image("Mention path for static folder /static/user_{}{}.png".format(j,i))


    def preprocess(self):

        self.access_data = self.data_read(self.access_file,self.target_col)
        
        for i in range(1,len(self.target_col)):
            if i == 1:
                self.access_data['ent_combo'] = self.access_data[self.target_col[i]]
            else:
                self.access_data['ent_combo'] = self.access_data['ent_combo'] + ' / ' + self.access_data[self.target_col[i]]

        self.access_data = self.access_data[[self.target_col[0],'ent_combo']]

        self.feature_engineering()

        self.cluster_results()

        self.access_data['labels'] = self.result_labels
        
        temp_mat = []
        for i in range(6):
            row_1=['Cluster {}'.format(i)]
            row_1.append(len(self.access_data[self.access_data['labels']==i]))
            row_1.append(self.access_data[self.access_data['labels']==i][self.target_col[0]].values)
            temp_mat.append(row_1)
        
        new_access_df = self.data_read(self.access_file,self.target_col)
        new_access_df = pd.merge(new_access_df,self.access_data[[self.target_col[0],'labels']],on = self.target_col[0])

        self.user_data = self.data_read(self.user_file,self.varified_cols)
        self.user_info = pd.merge(self.user_data,self.access_data[[self.target_col[0],'labels']],on = self.target_col[0])
        
        self.charts(new_access_df,self.user_info,6)

        remove = []
        for i in self.varified_cols[1:]:
            self.user_info[i] = self.user_info[i].astype('str')
            for j in self.user_info[i].unique():
                self.user_info[i+'_'+j] = np.where(self.user_info[i] == j,1,0)
            remove.append(i)
        self.user_info = self.user_info.drop(remove, axis=1)
        
        self.job_id = self.user_info.columns[2:]

        self.user_info['info_vector'] = self.user_info.iloc[:,2:].values.tolist()

        outlier_dict, self.outlier_uid = self.finding_outliers(6)

        self.outlier = self.user_info[self.user_info[self.varified_cols[0]].isin(self.outlier_uid)]
        self.access_outlier = self.access_data[self.access_data[self.varified_cols[0]].isin(self.outlier_uid)]
        self.user_info_wo_outlier = self.user_info[~self.user_info[self.varified_cols[0]].isin(self.outlier_uid)]
        self.access_info_wo_outlier = self.access_data[~self.access_data[self.varified_cols[0]].isin(self.outlier_uid)]
        
        self.relocating_outliers(6)
        
        self.entitlement_recommendation(6)
        self.vector_data(6)
        
        
        dummy_df = pd.read_csv(self.user_file, encoding='cp1252')
        result_data  = pd.merge(dummy_df, self.outlier[[self.varified_cols[0],'labels','new_labels','update','ent_remain','ent_remove','ent_add']], on=self.varified_cols[0])
        result_data.to_csv('Mention path for static folder /static/result_data.csv',index=False)


        return temp_mat
        


        




