# from curses.textpad import Textbox
#from curses import window
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import time
from turtle import color, width
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from lightgbm import LGBMRegressor, LGBMClassifier
import warnings

pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

#global vars
window1 = Tk()
window1.title('Critical Mass Project')
window1.geometry('300x150') 
trainingDir=''
predictionDir=''
labeled_data=pd.DataFrame
upper_funnel=pd.DataFrame
middle_funnel=pd.DataFrame
lower_funnel=pd.DataFrame

upper_funnel_2=pd.DataFrame
middle_funnel_2=pd.DataFrame
lower_funnel_2=pd.DataFrame
cat_cols=[[]]


def p2f(x):
  #change percentage to float
  if type(x) == str:
    return float(x.strip('%'))
  return x
def dollar2f(x):
  #change dollar string to float
  if type(x) == str:
    x=x.replace(',', '')
    return float(x.strip('$'))
  return x

# Define data cleaning functions
def rename_cols(df, s):
  return [s + str(num) for num in range(df.shape[1])]


def p2f(x):
  # Change percentage to float
  if type(x) == str:
      return float(x.strip('%'))
  return x


def dollar2f(x):
  # Change dollar string to float
  if type(x) == str:
      x = x.replace(',', '')
      return float(x.strip('$'))
  return x


def clean_percentage_df(df, s):
  df.columns = rename_cols(df, s)
  df = df.applymap(p2f)
  return df


def clean_nonpercentage_df(df, s):
  df.columns = rename_cols(df, s)
  df = df.applymap(dollar2f)
  return df

def dfMaker(df):
    j=0
    rowInd=0
    result=[[]]
    
    while(j<=13270):
        row=[]
        i=1
        while(i<=102):
            if(pd.notna( df.iloc[j,i])):
                row.append(df.iloc[j,i])
                print(row[rowInd])  
                rowInd+=1
            i+=1
        rowInd=0
        result.append(row)
        j+=1
    return pd.DataFrame(result)


def labeledDataMaker():
    data= pd.read_csv(trainingDir)
    cols=list(data.columns)
    data.head()
    # getting data from all collums except the last one
    x=data.iloc[:,0:18].values
    spend=data.iloc[:,18:120]
    imp=data.iloc[:,120:222]
    ctr=data.iloc[:,222:324]
    crvSV=data.iloc[:,324:426]
    crvBF=data.iloc[:,426:528]
    InterCVR=data.iloc[:,528:630]

    spend=spend.reset_index()
    imp=imp.reset_index()
    ctr=ctr.reset_index()
    crvSV=crvSV.reset_index()
    crvBF=crvBF.reset_index()
    interCVR=InterCVR.reset_index()

    # impressionDF=dfMaker(imp)
    # spendingDF=dfMaker(spend)
    # ctrDF=dfMaker(ctr)
    # cSVDF=dfMaker(crvSV)
    # cBFDF=dfMaker(crvBF)
    # internetDF=dfMaker(interCVR)
    data_ctr = clean_percentage_df(dfMaker(ctr), 'CTR_W')
    data_cvrBF = clean_percentage_df(dfMaker(crvBF), 'CVR(BF)_W')
    data_cvrSV = clean_percentage_df(dfMaker(crvSV), 'CVR(SV)_W')
    data_internet = clean_percentage_df(dfMaker(interCVR), 'Internet_CVR(S)_W')
    data_imps = clean_nonpercentage_df(dfMaker(imp), 'Impressions_W')
    data_spend = clean_nonpercentage_df(dfMaker(spend), 'Spend_W')

    # Combine weekly variables
    week_data_combined = pd.concat([data_ctr, data_cvrBF, data_cvrSV, data_internet, data_imps, data_spend], axis=1)
    week_data_combined.fillna(0, inplace=True)

    spends = [s for s in cols if s.endswith('Spend')]
    impression= [s for s in cols if s.endswith('Impressions')]
    CTR = [s for s in cols if s.endswith('CTR')]
    CVRS = [s for s in cols if s.endswith('CVR (S)')]
    CVRBF = [s for s in cols if s.endswith('CVR (BF)')]
    CVRSV = [s for s in cols if s.endswith('CVR (SV)')]

    sub=cols[0:18]
    kpi_means=[]
    for kpi in [CTR,CVRS,CVRBF,CVRSV]:
        df=data[kpi]
        df=df.applymap(p2f)
        mean_KPI=df.mean(axis=1,skipna=True)
        kpi_means.append(mean_KPI)
    spend_df=data[spends]
    spend_df=spend_df.applymap(dollar2f)
    mean_spend=spend_df.mean(axis=1,skipna=True)

    spend_df=data[spends]
    spend_df=spend_df.applymap(dollar2f)
    total_spend=spend_df.sum(axis=1,skipna=True)

    impr_df=data[impression]
    impr_df=impr_df.applymap(dollar2f)
    total_impressions=impr_df.sum(axis=1,skipna=True)


    # compute KPI max value
    kpi_peak=[]
    for kpi in [CTR,CVRS,CVRBF,CVRSV]:
        df=data[kpi]
        df=df.applymap(p2f)
        peak_KPI=df.max(axis=1,skipna=True)
        kpi_peak.append(peak_KPI)

    # compute KPI median values
    kpi_med=[]
    for kpi in [CTR,CVRS,CVRBF,CVRSV]:
        df=data[kpi]
        df=df.applymap(p2f)
        med_KPI=df.max(axis=1,skipna=True)
        kpi_med.append(med_KPI)
    new=data[sub]
    #label encoder
    labelencoder = LabelEncoder()

    categoricals = list(new.columns)
    unwanted_col = {'Creative ID', 'discount'}
    categoricals = [ele for ele in categoricals if ele not in unwanted_col]

    for i in categoricals:
        new[i] = labelencoder.fit_transform(new[i])

    #add kpi related columns
    new['Mean_Spend'] = mean_spend
    mean_names = ['Mean_CTR','Mean_CVRS','Mean_CVRBF','Mean_CVRSV']
    new['total_spend']=total_spend
    new['total_Impressions']=total_impressions
    peak_names=['Peak_CTR','Peak_CVRS', 'Peak_CVRBF', 'Peak_CVRSV']
    med_names=['Med_CTR', 'Med_CVRS', 'Med_CVRBF', 'Med_CVRSV']

    #generate new data set
    for i in range(len(mean_names)):
        new[mean_names[i]]=kpi_means[i]

    for j in range(len(peak_names)):
        new[peak_names[j]]=kpi_peak[j]

    for j in range(len(med_names)):
        new[med_names[j]]=kpi_med[j]
    new.head()

    # Grab features created to merge with dummy_data
    feat_created = new.iloc[:, 18:]

    

    # Combine everything
    global labeled_data
    labeled_data = pd.concat([new, week_data_combined], axis=1)
  
def setup():
    labeled_ml_data = pd.read_csv(r"labeled_ml.csv")
    raw_data = pd.read_csv(trainingDir)
    social_data = raw_data[raw_data.channel == 'Social'].drop(columns=['channel', 'size'])
    #Merging dataframes
    social_data = social_data.iloc[:,:16]

    to_drop = ['funnel', 'publisher', 'lob', 'product',	'theme', 'kpi_audience', 'creative_versions',
    'price', 'price_placement', 'discount', 'offer_placement', 'video_type', 'length']

    # Merge away outliers and concatenate target features
    ml_data_1 = pd.merge(social_data, labeled_ml_data.drop(to_drop, axis=1), left_on='Creative ID', right_on='Creative ID')
    ml_data_1.drop(['Creative ID', 'offer_group', 'asset_type'], axis=1, inplace=True)
    
    # Object to category
    global cat_cols
    cat_cols = ['funnel', 'publisher', 'lob', 'product', 'theme', 'kpi_audience', 'creative_versions',
    'price', 'price_placement', 'offer_placement', 'video_type', 'length']

    ml_data_1[cat_cols] = ml_data_1[cat_cols].apply(lambda x: x.astype('category'))
    ml_data_1[cat_cols] = ml_data_1[cat_cols].apply(lambda x: x.cat.codes.astype('category').astype('category'))
    #Seperate data by funnel
    global upper_funnel
    upper_funnel = ml_data_1[ml_data_1.funnel == 2]
    global middle_funnel
    middle_funnel = ml_data_1[ml_data_1.funnel == 1]
    global lower_funnel
    lower_funnel = ml_data_1[ml_data_1.funnel == 0]

    ml_data_2 = pd.read_csv(r"ml_all_data.csv")
    ml_data_2.drop(['Creative ID', 'offer_group', 'asset_type', 'size', 'discount'], axis=1, inplace=True)
    global upper_funnel_2
    upper_funnel_2 = ml_data_2[ml_data_2.funnel == 2]
    global middle_funnel_2
    middle_funnel_2 = ml_data_2[ml_data_2.funnel == 1]
    global lower_funnel_2
    lower_funnel_2 = ml_data_2[ml_data_2.funnel == 0]

def predictCTR():
    #Predict Average CTR
    X = upper_funnel.loc[:,['funnel', 'publisher', 'kpi_audience', 'video_type', 'theme', 'creative_versions', 'price']]
    y = upper_funnel.iloc[:, 14:]
    predictData=pd.read_csv(predictionDir)
   
    # Columns to encode
    one_hot = ['publisher', 'kpi_audience', 'theme', 'creative_versions']

    # Preprocessing
    ct = make_column_transformer(
        (OrdinalEncoder(), ['funnel']),
        (OneHotEncoder(handle_unknown='ignore'), one_hot),
        remainder='passthrough'
    )

    # Pipeline
    rf = RandomForestRegressor(n_estimators = 100, max_depth = 10, min_samples_leaf=15, random_state=123)
    pipe = make_pipeline(ct, rf)

    # Model evaluation
    scorers = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    cv = cross_validate(pipe, X, y['Mean_CTR'], cv=5, scoring=scorers, error_score='raise')
    r2 = cv.get('test_r2').mean().round(2)
    mse = -1*cv.get('test_neg_mean_squared_error').mean().round(3)
    mae = -1*cv.get('test_neg_mean_absolute_error').mean().round(3)
    print('Mean CTR')
    print(f'R2: {r2}\nMSE: {mse}\nMAE: {mae}\n')



    pipe.fit(X,y['Mean_CTR'])

    y_predict = pipe.predict(predictData)
    y_predict=rename_cols(pd.DataFrame(y_predict),'CTR_W')
    y_predict.to_csv("predictionCTR.csv")
    #print('These are predicted values: ', y_predict)
    print('These are the corresponding actual values: ')

    print(np.array(y['Mean_CTR']))

    # score=r2_score(y_test['Mean_CTR'], y_predict)
    # print("R2 is: ", score)
    # print("mean squared error is: ", mean_squared_error(y_test['Mean_CTR'], y_predict))
    # print("mean absolute error is: ", mean_absolute_error(y_test['Mean_CTR'], y_predict))
def predictorCVRS():
    #This is an LGBM model for predicting Sales CVR, it performs slightly better than RF 
    X = lower_funnel_2.iloc[:,:13]
    y = lower_funnel_2.iloc[:, 13:]
    predictData=pd.read_csv(predictionDir)

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Columns to impute and encode
    imp_cols = ['price_placement', 'offer_placement', 'video_type']
    one_hot = cat_cols[1:]

    # Preprocessing
    ct = make_column_transformer(
        (OrdinalEncoder(), ['funnel']),
        (OneHotEncoder(handle_unknown='ignore'), one_hot),
        remainder='passthrough'
    )

    # Pipeline
    lgbm = LGBMRegressor(learning_rate=0.025,  num_iterations= 400, min_child_samples=5, random_state=123)
    pipe = make_pipeline(ct, lgbm)

    # Model evaluation
    mean_cvrs = ['Mean_CVRS']
    scorers = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    for mean_kpi in mean_cvrs:
        cv = cross_validate(pipe, X_train, y_train[mean_kpi], cv=5, scoring=scorers, error_score='raise')
        r2 = cv.get('test_r2').mean().round(2)
        mse = -1*cv.get('test_neg_mean_squared_error').mean().round(4)
        mae = -1*cv.get('test_neg_mean_absolute_error').mean().round(4)
        print(mean_kpi)
        print(f'R2: {r2}\nMSE: {mse}\nMAE: {mae}\n')
        



    pipe.fit(X_train,y_train['Mean_CVRS'])
    y_predict = pipe.predict(predictData)
    print('These are predicted values: ', y_predict[0:5])

  
    y_predict=rename_cols(pd.DataFrame(y_predict),'CVR(SV)_W')
    y_predict.to_csv("predictionCVRSV.csv")

    print('These are the corresponding actual values: ')
    y_test1 = y_test['Mean_CVRS']
    print(np.array(y_test1[0:5]))

    # score=r2_score(y_test['Mean_CVRS'], y_predict)
    # print("R2 is: ", score)
    # print("mean squared error is: ", mean_squared_error(y_test['Mean_CVRS'], y_predict))
    # print("mean absolute error is: ", mean_absolute_error(y_test['Mean_CVRS'], y_predict))
def predictCVRSV():
    #Predict Average CVRSV
    X = middle_funnel.iloc[:,:14]
    y = middle_funnel.iloc[:, 14:]
    predictData=pd.read_csv(predictionDir)
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Columns to encode
    one_hot = ['publisher', 'kpi_audience', 'theme', 'creative_versions']

    # Preprocessing
    ct = make_column_transformer(
        #(SimpleImputer(strategy='constant', fill_value='missing', add_indicator=True), imp_cols),
        (OrdinalEncoder(), ['funnel']),
        (OneHotEncoder(handle_unknown='ignore'), one_hot),
        remainder='passthrough'
    )

    # Pipeline
    rf = RandomForestRegressor(n_estimators = 200, max_depth = 5,  min_samples_leaf=3 , random_state=123)
    pipe = make_pipeline(ct, rf)

    # Model evaluation
    scorers = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    # Site Visit CVR
    cv = cross_validate(pipe, X_train, y_train['Mean_CVRSV'], cv=5, scoring=scorers, error_score='raise')
    r2 = cv.get('test_r2').mean()
    mse = -1*cv.get('test_neg_mean_squared_error').mean()
    mae = -1*cv.get('test_neg_mean_absolute_error').mean()
    print('Mean Site Visit CVR')
    print(f'R2: {r2}\nMSE: {mse}\nMAE: {mae}\n')

    pipe.fit(X_train,y_train['Mean_CVRSV'])

    
    y_predict = pipe.predict(predictData)
    y_predict=rename_cols(pd.DataFrame(y_predict),'CVR(SV)_W')
    y_predict.to_csv("predictionCTR.csv")
    print('These are predicted values: ', y_predict[0:5])
    print('These are the corresponding actual values: ')

    print(np.array(y_test['Mean_CVRSV']))

    # score=r2_score(y_test['Mean_CVRSV'], y_predict)
    # print("R2 is: ", score)
    # print("mean squared error is: ", mean_squared_error(y_test['Mean_CVRSV'], y_predict))
    # print("mean absolute error is: ", mean_absolute_error(y_test['Mean_CVRSV'], y_predict))
def predictCVRBF():
    #Predict CVRBF
    X = middle_funnel_2.iloc[:,:13]
    y = middle_funnel_2.iloc[:, 13:]
    predictData=pd.read_csv(predictionDir)
    # Split train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Columns to encode
    one_hot = ['channel', 'publisher', 'kpi_audience', 'theme', 'creative_versions']

    # Preprocessing
    ct = make_column_transformer(
        #(SimpleImputer(strategy='constant', fill_value='missing', add_indicator=True), imp_cols),
        (OrdinalEncoder(), ['funnel']),
        (OneHotEncoder(handle_unknown='ignore'), one_hot),
        remainder='passthrough'
    )

    # Pipeline
    rf = RandomForestRegressor(n_estimators = 200, max_depth = 15, min_samples_leaf=5 , random_state=123)
    pipe = make_pipeline(ct, rf)

    # Model evaluation
    scorers = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    # Buy Flow CVR
    cv = cross_validate(pipe, X, y['Mean_CVRBF'], cv=5, scoring=scorers, error_score='raise')
    r2 = cv.get('test_r2').mean().round(2)
    mse = -1*cv.get('test_neg_mean_squared_error').mean().round(4)
    mae = -1*cv.get('test_neg_mean_absolute_error').mean().round(4)
    print('Mean Buy Flow Entry CVR')
    print(f'R2: {r2}\nMSE: {mse}\nMAE: {mae}\n')


    pipe.fit(X,y['Mean_CVRBF'])
    y_predict = pipe.predict(predictData)
    print('These are predicted values: ', y_predict[0:5])
    print('These are the corresponding actual values: ')

    print(np.array(y['Mean_CVRBF']))
    y_predict = pipe.predict(predictData)
    y_predict=rename_cols(pd.DataFrame(y_predict),'CVR(BF)_W')
    y_predict.to_csv("predictionCVRBF.csv")

    
#method to open the file explorer and get the file directory
def open_file(option):
    global trainingDir
    global predictionDir
    file_path = askopenfile(mode='r', filetypes=[('CSV Files', '*csv')])
    if file_path is not None:
        pass
    if(option):
        trainingDir=file_path.name
        print("The Training Data directory is: ",trainingDir)
        labeledDataMaker()
        setup()
    else:   
        predictionDir=file_path.name
        print("The desired Prediction Data directory is: ",predictionDir)



#button and label for the training data
trainingLabel= Label(window1, text='Upload a Training File ')
trainingLabel.grid(row=0, column=0, padx=10)
trainingButton = Button(window1, text ='Choose File', command = lambda:open_file(True)) 
trainingButton.grid(row=0, column=1)

#button and UI for the prediction data
predictionLabel= Label(window1, text='Upload a Prediction File ')
predictionLabel.grid(row=1, column=0, padx=10)
predictionButton = Button(window1, text ='Choose File', command = lambda:open_file(False)) 
predictionButton.grid(row=1, column=1)


#button and UI for the Prediction
predictorCTRLabel= Label(window1, text='Predict average CTR')
predictorCTRLabel.grid(row=2, column=0, padx=10)
predictorCTRButton = Button(window1, text ='Enter', command = lambda:predictCTR()) 
predictorCTRButton.grid(row=2, column=1)

#button and UI for the Prediction
predictorCVRSVLabel= Label(window1, text='Predict Average CVRSV')
predictorCVRSVLabel.grid(row=3, column=0, padx=10)
predictorCVRSVButton = Button(window1, text ='Enter', command = lambda:predictCVRSV()) 
predictorCVRSVButton.grid(row=3, column=1)

#button and UI for the Prediction
predictorCVRBFLabel= Label(window1, text='Predict Predict CVRBF')
predictorCVRBFLabel.grid(row=4, column=0, padx=10)
predictorCVRBFButton = Button(window1, text ='Enter', command = lambda:predictCVRBF()) 
predictorCVRBFButton.grid(row=4, column=1)

#button and UI for the Prediction
predictorCVRSLabel= Label(window1, text='Predict Predict CVRS')
predictorCVRSLabel.grid(row=5, column=0, padx=10)
predictorCVRSButton = Button(window1, text ='Enter', command = lambda:predictorCVRS()) 
predictorCVRSButton.grid(row=5, column=1)


window1.mainloop()