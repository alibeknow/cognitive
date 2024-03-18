import re
from tqdm import tqdm
from ftlangdetect import detect
import pandas as pd

def corpus_from_file(corpus_file, split_blocks = True):

    if '.txt' in corpus_file[-5:]:
        with open(corpus_file, encoding='utf-8', errors='ignore') as f:
            raw_lines = f.read()


    if split_blocks:
        corpus = raw_lines.split("===")


    else:
        corpus = ru_sent_tokenize(raw_lines)


    return corpus

def collect_data(corpus_files, split_blocks = True):

    corpi = []

    for f in tqdm(corpus_files, leave = True, position = 0, desc = "Collecting from local datafiles"):
        corpi.append(corpus_from_file(f, split_blocks = split_blocks))

    return corpi

def rep(m):
    s = m.group(1)
    return ' '.join(re.split(r'(?=[A-Z])', s))

def upr(m):
    s = m.group(1)
    return s.replace("#", "")

def clean_sent(t):

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

    t = re.sub("@[A-Za-z0-9_]+","", t)
    t = re.sub(r"http\S+|www.\S+", "", t)

    t = re.sub(r'#([A-Z0-9]{2,})', upr, t)

    t = re.sub(r'#(\w+)', rep, t).strip()

    t = re.sub(r"Isa", "is a", t)

    t = re.sub(r"\'", "'", t)

    t = re.sub(r"&amp;", "and", t)

    t = re.sub(emoji_pattern, '', t)

    return t

def get_summary(query, \
                ru_summary_path = "data/klara_info_ru.txt", \
                en_summary_path = "data/klara_info_en.txt"):

    summary_files = {}

    summary_files['ru'] = ru_summary_path
    summary_files['en'] = en_summary_path

    lang = detect(query)['lang']

    if lang == 'ru':
        summary = corpus_from_file(summary_files['ru'])
    elif lang == 'en':
        summary = corpus_from_file(summary_files['en'])
    else:
        summary = corpus_from_file(summary_files['en'])

    return summary

def get_QA_rows(filepath, include_question_rows = False):

    df = pd.read_excel(filepath)
    df = df.dropna()

    df_questions = df['Вопрос'].values.tolist()
    df_answers = df['Суть ответа'].values.tolist()

    QA_rows = []

    for i in range(len(df_questions)):
        if include_question_rows:
            QA_rows.append(str(df_questions[i]) + "\n " + str(df_answers[i]))
        else:
            QA_rows.append(str(df_answers[i]))

    return QA_rows