import pandas as pd
import json
from NLP_Models import TextMining as tm
from NLP_Models import openewfile as of
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pickle

#persiapan data
def cleanningtext(data,sentiment = True):
    fSlang = of.openfile(path = './NLP_Models/slangword')
    bahasa = 'id'
    stops, lemmatizer, tokenizer = tm.LoadStopWords(bahasa, sentiment = sentiment)
    sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
    SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}

    tqdm.pandas()
    
    data['text'] = data['text'].astype('str')
    data['text'] = data['text'].str.lower()
    data = data[~data.text.str.contains('unavailable')]
    print('------------------------------------------------------------------------------------')
    print('CLEANING DATA TEXT IS PROCESSING')
    data['cleaned_text'] = data['text'].progress_apply(lambda x : tm.cleanText(x,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, tokenizer=tokenizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 3))
    print('CLEANING PROCESS IS DONE')
    print('------------------------------------------------------------------------------------')
    print('HANDLING NEGATION IS PROCESSING')
    data['cleaned_text'] = data['cleaned_text'].progress_apply(lambda x : tm.handlingnegation(x))
    print('HANDLING NEGATION IS DONE')
    data = data[data['cleaned_text'].notna()]
    return data
#preprosecing data
def dataPreparingandProcesing(path_data_negative, path_data_positive):
    with open(f'{path_data_negative}', 'r', encoding='utf-8') as file:
        dataNegative = json.load(file)

    with open(f'{path_data_positive}', 'r', encoding='utf-8') as file:
        dataPositive = json.load(file)

    dfNegative, dfPositive = pd.json_normalize(dataNegative).iloc[0:100], pd.json_normalize(dataPositive).iloc[0:100]
    df = pd.concat([dfNegative,dfPositive])
    df = df[['text', 'Label']]
    df = cleanningtext(df,sentiment = True)
    return df

def matrixtermfreq(data):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1) #KALIAN HARUS PAHAM MAKSUD DARI max_df dan min_df. Dan boleh kalian ganti parameter ini
    tfs = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame(tfs.todense(), columns=feature_names)
    return df, vectorizer

def training_models(X_train, X_test, y_train, y_test, vectorizer, model_path, tokenizer_path):
    print('------------------------------------------------------------------------------------')
    print('TRAINING MODEL IS PROCESSING')
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', SVC(probability=True))
    ])

    parameters = {
        'clf__C': [1, 10, 100],  # Misalnya, tambahkan lebih banyak nilai atau tambah beberapa hyperparameter lainnya
        'clf__gamma': [0.1, 0.01]  # Misalnya, tambahkan lebih banyak nilai atau tambah beberapa hyperparameter lainnya
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score), #sklearn/matriks
        'precision': make_scorer(precision_score, average='macro', zero_division=1), #ANDA BISA UBAH DISINI (MICRO/MACRO)
        'recall': make_scorer(recall_score, average='macro', zero_division=1), #ANDA BISA UBAH DISINI (MICRO/MACRO)
        'f1_score': make_scorer(f1_score, average='macro', zero_division=1) #ANDA BISA UBAH DISINI (MICRO/MACRO)
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=scoring, cv=5,refit='f1_score', return_train_score=True)
    grid_search.fit(X_train, y_train)
    print('------------------------------------------------------------------------------------')
    print('TRAIN MODEL IS DONE')
    metrics_score_each_fold = pd.DataFrame(grid_search.cv_results_)

    best_model = grid_search.best_estimator_

    print('------------------------------------------------------------------------------------')
    print('EVALUATING MODEL IS PROCESSING')
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('------------------------------------------------------------------------------------')
    print('EVALUATING MODEL IS DONE')

    with open(f'{model_path}'+'/'+'best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)

    with open(f'{tokenizer_path}'+'/'+'tokenizer.pkl', 'wb') as tokenizer_file:
        pickle.dump(vectorizer, tokenizer_file)

    print('------------------------------------------------------------------------------------')
    print('MODEL AND TOKENIZER HAVE BEEN SAVED | ALL PROCESS ARE DONE')
    return {'metrics_train': metrics_score_each_fold, 'best_model_parms': best_model, 'metrics_test': report}


if __name__ == "__main__":
    path_data_negative, path_data_positive = 'C:/Users/Dell/Desktop/blajar ngoding/datmin baru/DATASET/negative.json', 'C:/Users/Dell/Desktop/blajar ngoding/datmin baru/DATASET/positive.json'
    df = dataPreparingandProcesing(path_data_negative, path_data_positive)

    print('\n'*3)
    print('------------------------------------------------------------------------------------')
    print('Data yang digunakan untuk proses pelatihan dan pengujian')
    print('\n'*1)
    print(df)
    print('------------------------------------------------------------------------------------')

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['Label'], test_size=0.2, random_state=42)
    matrixTermFreq =  matrixtermfreq(X_train)
    vectorizer, term_matrix = matrixTermFreq[1], matrixTermFreq[0]
    
    print('\n'*3)
    print('------------------------------------------------------------------------------------')
    print('Term-Matrix hasil TF-IDF')
    print('\n'*1)
    print(term_matrix)
    print('------------------------------------------------------------------------------------')
    
    print('\n'*3)
    training_report = training_models(X_train, X_test, y_train, y_test, vectorizer, 'C:/Users/Dell/Desktop/blajar ngoding/datmin baru/MODEL', 'C:/Users/Dell/Desktop/blajar ngoding/datmin baru/TOKENIZER')

    print('\n'*3)
    print('------------------------------------------------------------------------------------')
    print('MODEL PERFORMANCE ON GRID-SEARCH CV')
    print('\n'*1)
    print(training_report['metrics_train'])
    training_report['metrics_train'].to_csv('C:/Users/Dell/Desktop/blajar ngoding/datmin baru/METRICS/gridSearchcv.csv')
    print('\n'*1)
    print('------------------------------------------------------------------------------------')
    print('BEST MODEL CONFIGURATION')
    print('\n'*1)
    print(training_report['best_model_parms'])
    print('\n'*1)
    print('------------------------------------------------------------------------------------')
    print('MODEL PERFORMANCE ON TEST DATA')
    print('\n'*1)
    print(training_report['metrics_test'])

