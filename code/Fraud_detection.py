import pandas as pd
import numpy as np
import os
import logging
import warnings
import scipy.stats as st
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import metrics
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade, EasyEnsemble
from sklearn.tree import export_graphviz
from sklearn2pmml import sklearn2pmml
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.pipeline import PMMLPipeline
from graphviz import Source
from IPython.display import Image


# Configure Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def data_preprocessing(dfinal):

    """ This function is used for treating missing value data.
        if missing value in a variable is greater than 50 percent:
            Discard the variables from analysis
        else:
            Use generic imputation techniques like mean and mode
    """

    logger.info('Started Data Preprocessing')

    #Check for variables with missing data
    logger.info('Number of missing values in each column \n{0}'.format(dfinal.isnull().sum()))
    missing_df = pd.DataFrame(dfinal.isnull().sum()).reset_index()
    missing_df.columns = ['name', 'count']
    missing_df['perc'] = (missing_df.loc[:,'count']/len(dfinal))*100
    missing_df = missing_df.loc[missing_df['perc']>50,:].reset_index(drop = True)
    dfinal.drop(list(set(list(missing_df['name']))), axis = 1,inplace=True)
    dfinal.drop(['customer'], axis = 1,inplace=True)

    #List of categorical varibles and numerical variables.These lists will be further used for converting variables to categorical and numerical datatype.
    logger.debug('Starting generic imputation methods on all the variables')
    cat_cols =['age', 'gender', 'merchant', 'category', 'fraud']
    num_cols =['amount']

    #Convert columns mentioned in cat_cols list to categorical variables
    logger.debug('Generic imputation on categorical variables')
    for var in cat_cols:
        dfinal[var] = dfinal[var].astype('category')
        if (var != 'fraud'):
            dfinal[var].fillna(dfinal[var].mode()[0], inplace=True)

    #Convert columns mentioned in num_cols list to numerical variables
    logger.debug('Generic imputation on numerical variables')
    for var in num_cols:
        dfinal[var].fillna(dfinal[var].mean(), inplace=True)
        dfinal[var] = dfinal[var].astype('int')

    return dfinal, cat_cols

def process_dataframe(input_df, encoder_dict=None):
    
    # Label encode categorical variables
    logger.info('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'category']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat])
        encoder_dict[feat] = encoder
    logger.info('Label encoding complete.')
    
    return input_df, categorical_feats.tolist(), encoder_dict

def print_chisquare_result(Dfinal, colX, alpha, p):

    result = ""
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(colX)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        del Dfinal[colX]

    logger.info(result)

    return Dfinal

def TestIndependence(Dfinal, colX, colY, alpha=0.05):

    """Implement Chi-Square Test between categorical variables"""
    logger.debug('Implementing Chi-Square test between columns {0} and {1}'.format(colX, colY))
    p = None #P-Value
    chi2 = None #Chi Test Statistic
    dof = None
    dfTabular = None
    dfExpected = None
    X = Dfinal[colX].astype(str)
    Y = Dfinal[colY].astype(str)
    logger.debug(Dfinal[colY].unique())
    dfObserved = pd.crosstab(Y,X)
    chi2, p, dof, expected = st.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    Dfinal = print_chisquare_result(Dfinal, colX, alpha, p)
    return Dfinal

def Balance_classes(X_train, y_train, Sampling_Function):
    if Sampling_Function == 'RandomUnderSampler':
        us = RandomUnderSampler(ratio = 0.5, random_state = 1)
    elif Sampling_Function == 'NearMiss1':
        us = NearMiss(ratio = 0.5, random_state = 1, version = 1, size_ngh = 3)
    elif Sampling_Function == 'NearMiss2':
        us = NearMiss(ratio = 0.5, random_state = 1, version = 2, size_ngh = 3)
    elif Sampling_Function == 'NearMiss3':
        us = NearMiss(ratio = 0.5, random_state = 1, version = 3, ver3_samp_ngh = 3)
    elif Sampling_Function == 'CondensedNearestNeighbour':
        us = CondensedNearestNeighbour(random_state = 1)
    elif Sampling_Function == 'EditedNearestNeighbours':
        us = EditedNearestNeighbours(random_state = 1, size_ngh = 5)
    elif Sampling_Function == 'RepeatedEditedNearestNeighbours':
        us = EditedNearestNeighbours(random_state = 1, size_ngh = 5)        
    elif Sampling_Function == 'TomekLinks':
        us = TomekLinks(random_state = 1)
    elif Sampling_Function == 'RandomOverSampler':
        us = RandomOverSampler(ratio = 0.5, random_state = 1)
    elif Sampling_Function == 'SMOTE':
        us = SMOTE(ratio = 0.5, k =5, random_state = 1)
    elif Sampling_Function == 'SMOTETomek':
        us = SMOTETomek(ratio = 0.5, k =5, random_state = 1)
    elif Sampling_Function == 'SMOTEENN':
        us = SMOTEENN(ratio = 0.5, k =5, random_state = 1, size_ngh = 5)
    elif Sampling_Function == 'EasyEnsemble':
        us = EasyEnsemble()
    elif Sampling_Function == 'BalanceCascade_rf':
        us = BalanceCascade(classifier = 'random-forest', random_state = 1)
    elif Sampling_Function == 'BalanceCascade_svm':
        us = BalanceCascade(classifier = 'linear-svm', random_state = 1)        

    X_train_res, y_train_res = us.fit_sample(X_train, y_train)

    return X_train_res, y_train_res



def load_data(foldername):
    
    """ Read input data """
    input_dir = os.path.join(os.pardir, foldername)
    logger.info('Input files:\n{}'.format(os.listdir(input_dir)))
    logger.info('Loading data sets...')
    sample_size = None
    Transactional_df = pd.read_csv(os.path.join(input_dir, 'Transactional_dataset.csv'), nrows=sample_size).reset_index()
    Network_df = pd.read_csv(os.path.join(input_dir, 'Network_dataset.csv'), nrows=sample_size).reset_index()
    logger.info('Data loaded.')
    logger.info('Main application Transactional data set shape = {}'.format(Transactional_df.shape))
    logger.info('Main application Network data set shape = {}'.format(Network_df.shape))

    return Transactional_df, Network_df

def DecisionTree(X_train, y_train, X_train_res, y_train_res, X_test, y_test):
    
    """ Implementing Decision Tree """
    dtc = DecisionTreeClassifier(random_state=0, class_weight="balanced",max_leaf_nodes = 10)
    X = pd.DataFrame(X_train_res)
    X.columns = X_train.columns
    y = pd.DataFrame(y_train_res)
    y.columns = ['fraud']
    dtc.fit(X,y)
#    pipeline_model = PMMLPipeline([('dtc',dtc)]).fit(X, y)
    input_dir = os.path.join(os.pardir, 'code')
#    sklearn2pmml(pipeline_model,os.path.join(input_dir,"FraudClassificationTree.pmml"), with_repr = True)

    logger.info('Implementing Decision tree V1')
    logger.info('Accuracy on the training subset: {:.3f}'.format(dtc.score(X_train_res, y_train_res)))
    logger.info('Accuracy on the test subset: {:.3f}'.format(dtc.score(X_test, y_test)))
        
    """ Feature Importance for DT """
    n_features = X_train_res.shape[1]
    plt.barh(range(n_features), dtc.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.rcParams["figure.figsize"] = (7,5)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_imp_dt')
    plt.show()
    
    """ Generate predictions on test data using DT model """
    DT_predictions = dtc.predict(X_test)
    DT_predictions_proba = dtc.predict_proba(X_test)[:,1]
    logger.debug('DT_PREDICTIONS are as follows \n:{0}'.format(DT_predictions))
    
    """ Confusion Matrix for DT """
    confusion_mat = confusion_matrix((y_test), DT_predictions)
    logger.debug('Classification_report is : \n {0}'.format(classification_report((y_test), DT_predictions)))
    logger.debug('Confusion matrix is \n {0}'.format(confusion_mat))
    logger.debug('Ratio of classes before implementing sampling method: \n {0}'.format(Counter(y_train)))
    logger.debug('Ratio of classes after implementing sampling method: \n {0}'.format(Counter(y_train_res)))
    
    """ ROC Curve for DT """
    fpr, tpr, _ = metrics.roc_curve(pd.to_numeric(y_test), DT_predictions_proba.round())
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.rcParams["figure.figsize"] = (7,5)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('DT_ROC')
    plt.show()
    
    graph = Source(export_graphviz(dtc, out_file=None, feature_names=X_train.columns))
    png_bytes = graph.pipe(format='png')
    with open('dtree_pipe.png','wb') as f:
        f.write(png_bytes)
    Image(png_bytes)    

    return


def main():

    sampling_func = 'SMOTE'

    """ Load Data set """
    foldername = 'data'
    Transactional_df, Network_df = load_data(foldername)

    Transactional_df['common_id'] = Transactional_df['index'].astype('str') + Transactional_df['customer'].astype('str') + Transactional_df['merchant'].astype('str') + Transactional_df['category'].astype('str') + Transactional_df['fraud'].astype('str')
    Network_df['common_id'] = Network_df['index'].astype('str') + Network_df['Source'].astype('str') + Network_df['Target'].astype('str') + Network_df['typeTrans'].astype('str') + Network_df['fraud'].astype('str')

    """ Merge Transactional and Network dataset into single DataFrame """
    dfinal = Transactional_df[['common_id', 'step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount']].merge(Network_df, how='inner', on = ['common_id'])
    dfinal = dfinal[['customer', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']]
    logger.debug('List of all the variables :\n{0}'.format(dfinal.columns))
    logger.debug('Shape of final data :\n{0}'.format(dfinal.shape))

    """ Data Preprocessing"""
    Dfinal,categorical_feats = data_preprocessing(dfinal)

    """ Feature Selection"""
    logger.debug('Starting Feature Selection')
    logger.debug('Implementing Chi Square Test between dependent variable "FRAUD" and all other categorical variables')

    """ Chi Square Test """
    for var in categorical_feats:
        if (var != 'fraud'):
            Dfinal = TestIndependence(Dfinal,colX=var, colY='fraud')

    """ Process the data set """
    merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=Dfinal)
    logger.debug('Shape of final data :\n{0}'.format(merged_df.head()))

    """ Train the model """
    logger.debug('Start Model Training')
    merged_df['fraud'] = merged_df['fraud'].astype(str)
    fraud = merged_df.pop('fraud')
    
    """ Train and test split """
    logger.debug('Train test split')
    X_train, X_test, y_train, y_test = train_test_split(merged_df,fraud, test_size=0.3, random_state=0)
    x = y_train.to_frame(name='fraud')
    
    """ Handle data Imbalance """
    logger.info('Ratio of target variable classes:\n{}'.format(x['fraud'].value_counts()))
    logger.info('Sampling Function being used for handling imbalanced data: {0}'.format(sampling_func))
    X_train_res, y_train_res = Balance_classes(X_train, y_train, Sampling_Function = sampling_func)

    """ Implementing Decision Tree """
    DecisionTree(X_train, y_train, X_train_res, y_train_res, X_test, y_test)

if __name__ == '__main__':
    main()
