from classes import *
import seaborn as sns
import random
import pickle


sns.set(rc={'figure.figsize': (11.7, 8.27)}, style="darkgrid")


def make_data(threshold_for_dropping=0):
    encounter_data = pd.read_csv("encounter.csv").rename(str.lower, axis='columns')
    data = encounter_data.loc[:, ['soap_note', 'cc']]
    data.dropna(inplace=True, subset=['soap_note'])
    data.reset_index(drop=True, inplace=True)
    data['cc'].fillna('no specific issue', inplace=True)
    data['cc'] = data['cc'].str.lower()
    if threshold_for_dropping > 0:
        temp_dict = data['cc'].value_counts().to_dict()
        temp_list = [index for index, rare_labels in enumerate(data['cc'].values)
                     if temp_dict[rare_labels] <= threshold_for_dropping]
        data.drop(temp_list, inplace=True)
        data.reset_index(drop=True, inplace=True)
    data.sort_values('cc',inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def preprocess_data(data):
    soap = data['soap_note']
    soap_temp = [re.split('o:''|''o :', i) for i in soap]  # split by "o:" or "o :"
    temp_sentences = [i[0].strip().strip('s:').lower() for i in soap_temp]
    try:
        _ = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download('stopwords')
    stopword_set = set(stopwords.words("english"))
    document_ = Document('\n '.join(temp_sentences))
    document_.do_replaces()
    document_.make_sentences('\n ')
    for index,sentence in enumerate(document_.sentences):
        sentence.label=data['cc'][index]
        sentence.stem_and_check_stop(stopword_set)
        sentence.make_tokens()
        sentence.make_original_text_tokens()
        sentence.text = ' '.join(sentence.tokens)
    document_.train_test_split()
    document_.train.make_lexicon()
    document_.make_dict()
    pickle.dump(document_.labels_dict,open("labels_dict.pkl", "wb"))
    print('Classes are ready to use\n')
    return document_