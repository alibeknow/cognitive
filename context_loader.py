from methods.text_processors import *
from methods.dpr import *

verbose = True

QA_rows = get_QA_rows("data/Q&A.xlsx", include_question_rows = True)

additional_info = collect_data(["data/gen_book_desc_en.txt"], split_blocks = True)

def RAG_inference(query, n_answers = 2, thresh = 0.25, char_lim = 8192, \
                    bump_qa = False, qa_coef = 1.15, verbose = False, logger = None):

    summary = get_summary(query, ru_summary_path = "data/klara_info_ru.txt", \
                                    en_summary_path = "data/klara_info_en.txt")
    corpi = [summary, QA_rows]
    corpi.extend(additional_info)

    if bump_qa:
        qa_idx = 1
    else:
        qa_idx = None

    context = DPR_answer_query(query, corpi, \
                        n_answers = n_answers, thresh = thresh, char_lim = char_lim, \
                        qa_idx = qa_idx, qa_coef = qa_coef, verbose = verbose,
                        logger = logger)

    if verbose:
        if logger is not None:
            logger.trace("Found potentially relevant context: %s\n" % context)
        else:
            print("\nFound potentially relevant context: %s\n" % context)

    return context

prompt_template = """System: Dialogue between curious User and Klara AI Assistant, a helpful and friendly female AI Assistant who can communicate in English and Russian and mimic user's language and style of writing. Regardless of what the previous answers and questions were, Klara AI will now generate a reply strictly in {user_language}. Klara AI, or Klara for short, is an intelligent assistant (who acts and speaks as a woman) within the app of the same name, and a digital neo-bank that provides various financial services to women in an interactive, innovative way.
Klara AI answers the User's questions in an informal, concise way, yet the answers are always accurate, and have enough detail. When it comes to the product or Klara's financial services, Klara AI bases her answer only on the provided Context, and does not give information that she is unconfident about. Klara AI will never attempt to do any calculations or currency conversions, or provide answers that require external, up-to-date knowledge that is not otherwise mentioned in the context. Klara AI answers in the style and tone of User's messages, and adjusts the style of her answer to User's age (acts as if Klara AI Assistant was {user_age}!). Klara AI knows that User's name is {user_name}, {pronoun} is {user_age} years old, and {pronoun} is {user_gender}. Klara AI tries to never repeat herself, and be concise and to the point. Background info: Klara AI is based in Kazakhstan, where the main currency is Tenge (KZT), and a PIN-code is always a 4-digit number (no letters).

Context: {context}

Conversation history:
{few_shot}
{history}
User: {input}
Klara AI:"""