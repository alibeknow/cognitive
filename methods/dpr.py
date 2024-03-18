from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from rusenttokenize import ru_sent_tokenize
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

if torch.cuda.is_available():
    if torch.cuda.device_count() == 2:
        device = "cuda:0"
    elif torch.cuda.device_count() == 1:
        device = "cuda:0"
else:
    device = "cpu"

embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device = device)

def detect_relevant_corpus(query, corpi):

    assert len(corpi) >= 1

    corpi_embeddings = embedder.encode(corpi, convert_to_tensor = True)
    query_embeddings = embedder.encode(query, convert_to_tensor = True)

    cos_scores = util.cos_sim(query_embeddings, corpi_embeddings)[0]

    top_results = torch.topk(cos_scores, k = 1)

    idx = top_results[1].item()

    return corpi[idx], idx

def DPR_answer_query(query, corpi, \
                    n_answers = 2, thresh = 0.25, char_lim = 8192, \
                    qa_idx = None, qa_coef = 1.25, verbose = False, recursive = False, \
                    logger = None):

    if logger is not None:
        logger.trace("RAG model: using device %s" % device)
    else:
        print("\nUsing device %s\n" % device)

    corpus, corpus_idx = detect_relevant_corpus(query, corpi)

    if qa_idx and not recursive:
        if len(corpi) >= qa_idx and corpi[qa_idx] != corpus:
            corpus.extend(corpi[qa_idx])

    if not corpus:
        return ""

    top_k = min(n_answers, len(corpus))

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor = True)
    query_embeddings = embedder.encode(query, convert_to_tensor = True)

    cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)[0]
    if qa_idx:
        cos_scores[qa_idx] *= qa_coef

    top_results = torch.topk(cos_scores, k = top_k)

    if n_answers == 2:

        if top_results[0][0].item() > thresh and top_results[0][1].item() > thresh:
            answer = "\n ".join([corpus[idx] for idx in top_results[1]]).replace("..", ".")

        elif top_results[0][0].item() > thresh:
            answer = corpus[top_results[1][0]]
        else:
            answer = ""

    else:

        if top_results[0][0].item() > thresh:
            answer = corpus[top_results[1][0]]
        else:
            answer = ""

    if answer and len(answer) > char_lim:

        if n_answers > 1:
            answer_cand = corpus[top_results[1][0]]

        if len(answer_cand) > char_lim:
            sents = ru_sent_tokenize(answer)
            lens = []
            for sent in sents:
                lens.append(len(sent))

            idx = np.argmin(lens)

            answer = sents[idx]

        else:
            answer = answer_cand

    if not answer:
        new_corpi = [corpi[i] for i in range(len(corpi)) if i != corpus_idx]
        if new_corpi:
            answer = DPR_answer_query(query, new_corpi, n_answers = n_answers, \
                                    thresh = thresh, char_lim = char_lim, \
                                    qa_idx = qa_idx, qa_coef = qa_coef, \
                                    recursive = True)
        else:
            answer = ""

    if verbose:
        score_1 = cos_scores[0].item()
        score_2 = cos_scores[1].item()

        if logger is not None:
            logger.trace("Context scores: %0.2f, %0.2f" % (score_1, score_2))
        else:
            print("\nContext scores: %0.2f, %0.2f" % (score_1, score_2))

    return answer