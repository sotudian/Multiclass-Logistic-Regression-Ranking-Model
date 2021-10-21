#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""


import pandas as pd
import numpy as np
import math
import os
from itertools import combinations 
import matplotlib.pyplot as plt
import collections
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Data Preprocessing
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Load data
CellDrug_Blood = pd.read_csv('/Users/BigData_CellDrug_Blood_normilized.csv')
CellDrug_Blood=CellDrug_Blood.drop(columns=['Index']) 

CellGene_Blood = pd.read_csv('/Users/BigData_CellGene_Blood_normilized.csv')
CellGene_Blood=CellGene_Blood.drop(columns=['Index']) 

Num_Drugs=50
Num_Genes=30
Train_X=CellGene_Blood.iloc[0:51,0:Num_Genes]
Train_Y=CellDrug_Blood.iloc[0:51,0:Num_Drugs]

Test_X=CellGene_Blood.iloc[51:,0:Num_Genes]
Test_Y=CellDrug_Blood.iloc[51:,0:Num_Drugs]



def Rank_Label(A):
    if list(A)==[1,2,3]:
        Label=[1,0,0,0,0,0]
    elif list(A)==[1,3,2]:
        Label=[0,1,0,0,0,0]
    elif list(A)==[2,1,3]:
        Label=[0,0,1,0,0,0]
    elif list(A)==[2,3,1]:
        Label=[0,0,0,1,0,0]
    elif list(A)==[3,1,2]:
        Label=[0,0,0,0,1,0]
    elif list(A)==[3,2,1]:
        Label=[0,0,0,0,0,1]
    
    return Label

 
All_comb = list(combinations(range(Train_Y.shape[1]), 3) )  # Get all combinations for drugs


X_Train_Com=[]
Y_Train_Com=[]

# Create one-hot for all drugs
OneHOT = list()
for rr in range(len(All_comb)):
	letter = [0 for x in range(Train_Y.shape[1])]
	letter[All_comb[rr][0]],letter[All_comb[rr][1]],letter[All_comb[rr][2]]= 1,1,1
	OneHOT.append(letter)
    
    
# Data generation - Triples - TRAIN
for q in range(Train_X.shape[0]):
    S_in=Train_X.iloc[q,:].values
    X1=[]
    X2=[]
    Y2=[]
    for ac in range(len(All_comb)):
        
        S_out=Train_Y.iloc[q,list(All_comb[ac])]
        S_out_Rank= S_out.rank(method = 'first',ascending=True)   # Rank drug descending - best smaller
        
        X1.append(list(S_in))    
        Y2.append(Rank_Label(S_out_Rank))
     
    X2=[x+y for x,y in zip(X1,OneHOT)]
        
    X_Train_Com=X_Train_Com+X2
    Y_Train_Com=Y_Train_Com+Y2
    

   
# Delete redundant VARs
del CellGene_Blood, CellDrug_Blood, S_in, S_out, S_out_Rank
del  X1,X2, Y2, ac, letter, q, rr, 


# Train validation split

TV_Ratio=0.99
TrainX,TrainY = np.array(X_Train_Com[0:round(len(X_Train_Com)*TV_Ratio)]) ,     np.array(Y_Train_Com[0:round(len(X_Train_Com)*TV_Ratio)])
ValidationX, ValidationY    = np.array(X_Train_Com[round(len(X_Train_Com)*TV_Ratio):]),     np.array(Y_Train_Com[round(len(X_Train_Com)*TV_Ratio):])

del X_Train_Com,Y_Train_Com,Train_X,Train_Y


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                   Mult-class LOGISTIC REGRESSION MODEL
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


from scipy.optimize import minimize  # optimization code
import matplotlib.pyplot as plt  # plotting
import seaborn as sns
sns.set()
import itertools  # combinatorics functions for multinomial code
import tensorflow as tf
from scipy.stats import spearmanr

# Functions
def softmax(u):
    expu=np.exp(u)
    return expu/np.sum(expu)

def crossEntropy(p,q):
    return -np.vdot(p, np.log(q))

def eval_L(X,Y,beta):
    L=0.0
    N = X.shape[0]
    for i in range(N):
        Xi_Hat = X[i]
        Yi = Y[i]
        qi = softmax( beta @ Xi_Hat )
        L += crossEntropy(Yi,qi)
    return L
    
def logReg_SGD(X,Y,alpha,numEpochs):
    
    N,d= X.shape
    X = np.insert(X,0,1,axis = 1)
    K= Y.shape[1]
    
    beta = np.zeros((K,d+1))
    Lvals= []
    for ep in range(numEpochs):
        L=eval_L(X,Y,beta)
        Lvals.append(L)
        print("Epoch Number " + str(ep+1) + ":    cost is  " + str(L))
        
        
        prm = np.random.permutation(N)
        for i in prm:
            Xihat= X[i]
            Yi = Y[i]
            
            qi = softmax( beta @ Xihat )
            Grad_Li = np.outer(qi-Yi,Xihat )
            beta = beta - alpha * Grad_Li
    return beta,Lvals

def PredictLabels(X,beta):
    All_Zeros= np.zeros(6)
    X = np.insert(X,0,1,axis = 1)
    N=X.shape[0]
    preictions=[]
    for i in range(N):
       Xihat= X[i] 
       qi = softmax( beta @ Xihat )
       # print(qi)      # Probability vector
       k= np.argmax(qi)
       
       preictions.append(k)
    return preictions

def Score_Calculator(label):
    # label={1,2,3,4,5,6} -1
    if label==0:
        Points=[3,2,1]
    elif label==1:
        Points=[3,1,2]
    elif label==2:
        Points=[2,3,1]
    elif label==3:
        Points=[1,3,2]
    elif label==4:
        Points=[2,1,3]
    elif label==5:
        Points=[1,2,3]
    return Points


def Performance_Metrics(True_Values,True_Rank,Predicted_Rank,Percebtile_Sensetivity):
    PERCENTILE = np.percentile(True_Values, Percebtile_Sensetivity)
    Sensetives_Drugs=[i for i in range(len(True_Values)) if True_Values[i]<=PERCENTILE]
    Predicted_sensetive_Rank = list(np.array(Predicted_Rank)[Sensetives_Drugs])
    Rel_Binary=np.zeros(len(True_Values))
    Rel_Binary[Sensetives_Drugs]=1

    # Spearman
    Spearman=spearmanr(Predicted_Rank, True_Rank)[0]
 
    # AH@5
    AH5=len(set(Predicted_sensetive_Rank)&set([i+1 for i in range(5)]))   

    # AH@10
    AH10=len(set(Predicted_sensetive_Rank)&set([i+1 for i in range(10)]))   

    return [Spearman,AH5,AH10]

                
                



# Training
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")  
alpha=0.001
numEpochs= 4
Num_sensetives=10


beta,Lvals=logReg_SGD(TrainX,TrainY,alpha,numEpochs)
plt.semilogy(Lvals)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")  

        
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                   Testing - Ranking
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    
RankingResults=[]
for q in range(Test_X.shape[0]):
    S_in=Test_X.iloc[q,:].values
    X1=[]
    X2=[]
    TestY=[]
    for ac in range(len(All_comb)):
        
        S_out=Test_Y.iloc[q,list(All_comb[ac])]
        S_out_Rank= S_out.rank(method = 'first',ascending=True)   # Rank drug descending - best smaller
        
        X1.append(list(S_in))    
        TestY.append(Rank_Label(S_out_Rank))
    
    TestX=[x+y for x,y in zip(X1,OneHOT)] # Test data - one patient
    Pred_Q= PredictLabels(TestX,beta)     # Prediction 
    
    SCORES_Q=np.zeros(Test_Y.shape[1])   # Scores vector # drugs 
        
    
    # Calculate scores for diffrent combinations
    
    for ss in range(len(All_comb)):
        Scores_seperate=[]
        Scores_seperate=Score_Calculator(Pred_Q[ss])
        #print(Scores_seperate)
        #print(All_comb[ss])
        # Assign scores
        SCORES_Q[All_comb[ss][0]]+= Scores_seperate[0]
        SCORES_Q[All_comb[ss][1]]+= Scores_seperate[1]
        SCORES_Q[All_comb[ss][2]]+= Scores_seperate[2]
        # print(SCORES_Q)
    print("Scores:  "+str(SCORES_Q))
    print("Predicted Rank:  "+str(list((pd.DataFrame(SCORES_Q).rank(method = 'first',ascending=False))[0])))  # Best Higher
    
    Test_True_LAbel=pd.DataFrame(Test_Y.iloc[q,:].values)
    Test_True_LAbel_Rank=Test_True_LAbel.rank(method = 'first',ascending=True)   # Rank drug descending - best smaller
    print("True Rank:  "+ str(list(Test_True_LAbel_Rank[0])))
    
    
    # Find ties
    print([item for item, count in collections.Counter(SCORES_Q).items() if count > 1])
    aa=[item for item, count in collections.Counter(SCORES_Q).items() if count > 1]
    if len(aa)>0:
        for u in aa:
            # print(sum(SCORES_Q==u))
            # print(list(np.where(SCORES_Q==u)[0]))
            
            if len(list(np.where(SCORES_Q==u)[0]))==2:        # 2 Ties     +++===----------------------------
    
                Index_2Tie=(list(np.where(SCORES_Q==u)[0]))                      # index of 2 ties
                Colms_2Tie=[x - (Test_Y.shape[1]) for x in list(Index_2Tie)]     # index of 2 ties from back        
                NP_TestX=np.array(TestX)                                       # Convert to NP  
                colList = [ i for i in range(len(TestX)) if all(NP_TestX[i,Colms_2Tie]) ]   # indices of com contain 2 ties
                for uu in colList:
                    Scores_2Tie=[]
                    Scores_2Tie=Score_Calculator(Pred_Q[uu])
                    # print(Scores_2Tie)
                    # print(All_comb[uu])
                    
                    Place_1_Tie=All_comb[uu].index(Index_2Tie[0])      # Find place(index) of ties 
                    Place_2_Tie=All_comb[uu].index(Index_2Tie[1])      # Find place(index) of ties
                    SCORES_Q[All_comb[uu][Place_1_Tie]]  +=   Scores_2Tie[Place_1_Tie]*0.001
                    SCORES_Q[All_comb[uu][Place_2_Tie]]  +=   Scores_2Tie[Place_2_Tie]*0.001
                   
                    
            elif len(list(np.where(SCORES_Q==u)[0]))==3:      # 3 Ties        +++===----------------------------
                Index_3Tie=(list(np.where(SCORES_Q==u)[0])) 
                OneHot_Tie=np.zeros(Test_Y.shape[1])   
                OneHot_Tie[Index_3Tie]=1
                Tie_X=list(S_in)+list(OneHot_Tie)
                Scores_3Tie=[]
                Scores_3Tie=Score_Calculator(Pred_Q[TestX.index(Tie_X)])   # Pred for 3-ties + Scores
                SCORES_Q[Index_3Tie[0]]+= Scores_3Tie[0]*0.001
                SCORES_Q[Index_3Tie[1]]+= Scores_3Tie[1]*0.001
                SCORES_Q[Index_3Tie[2]]+= Scores_3Tie[2]*0.001
                
            elif len(list(np.where(SCORES_Q==u)[0]))>3:       # More than 3 Ties       +++===----------------------------
                print("WARNING: More than 3 equal score")
                os.system( "say beep" )
        
    # Print new rankiong        
        print("New Scores:  "+str(SCORES_Q))
        print("New Predicted Rank:  "+str(list((pd.DataFrame(SCORES_Q).rank(method = 'first',ascending=False))[0]))) # Best Higher
        print("True Rank:           "+ str(list(Test_True_LAbel_Rank[0])))
    
    
    
    else:
        print("No Ties")    
        
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
   
    Percebtile_Sensetivity=round((Num_sensetives/Test_Y.shape[1])*100)
   
    True_Values=list(Test_True_LAbel[0])
    True_Rank=list(Test_True_LAbel_Rank[0])
    Predicted_Rank=list((pd.DataFrame(SCORES_Q).rank(method = 'first',ascending=False))[0])
    
    [Spearman,AH5,AH10]=Performance_Metrics(True_Values,True_Rank,Predicted_Rank,Percebtile_Sensetivity)
        
    print("# Spearman: ",Spearman, "# AH@5: ",AH5 , "AH@10: ",AH10)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")                
                    
    RankingResults.append([Spearman,AH5,AH10]) 



print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
Final_Results= np.sum(  np.array(RankingResults)  , 0) / float(len(RankingResults))  
print("Final Results: ")
print("1) Spearman: " ,Final_Results[0])
print("2)     AH@5: " ,Final_Results[1])
print("3)    AH@10: " ,Final_Results[2])
print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")





    
    
    
    
    
    
    
    
    