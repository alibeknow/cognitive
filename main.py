import langchain_community
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache
from ftlangdetect import detect
import hashlib
import uvicorn
from cust_vLLM import CustVLLM
from context_loader import RAG_inference, prompt_template
from loguru import logger
import argparse
import sys
import fasttext

fasttext.FastText.eprint = lambda x: None

parser = argparse.ArgumentParser(
                    prog='Klara AI',
                    description='LLM-powered bot pipeline for Klara AI')

# llama2 or mixtral (default: llama2)
parser.add_argument('-model', '--model', default = "llama2")
# llama2 or mixtral (default: llama2)
parser.add_argument('-model_path', '--model_path', default = "")
# for optional outputs (default: True)
parser.add_argument('-v', '--verbose', action = 'store_false')
# to redirect main logs to app.log file (default: False)
parser.add_argument('-lf', '--log_to_file', action = 'store_true')
# dialogue window size (default: 15)
parser.add_argument('-w', '--window', default = 15)

args = parser.parse_args()

logger.remove(0)
verbose = args.verbose
log_to_file = args.log_to_file
dialogue_window = args.window

if log_to_file:
    logger.add("app.log", level="TRACE", serialize = True)
else:
    logger.add(sys.stdout, level="TRACE", serialize = False)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )

langchain_community.llm_cache = GPTCache(init_gptcache)

"""
Main hyperparams:
    Temperature:
        Controls 'randomness' and creativity.
        Higher values lead to more chaotic - but sometimes more interesting - outputs
        Lower values lead to more stable and predictable outputs.
    Top_p:
        Controls percentage of available tokens sorted by their scores.
        Higher values -> higher threshold for selecting a pool of tokens with a high score
        Lower values -> more tokens are available for generation.
"""

if args.model == 'mixtral':

    model_path = "/m2/Models/hub/models--TheBloke--openbuddy-mixtral-8x7b-v15.2-AWQ/snapshots/b13a4c56e6a33d268a2329c8e8171971e7f9c823" # Mixtral-8x7B (~56B param model)

    llm = CustVLLM(model=model_path,
            trust_remote_code=True,
            max_new_tokens=512,
            temperature=0.4,
            top_p=0.6,
            vllm_kwargs={"quantization": "awq",
                            "gpu_memory_utilization": 0.70,
                            "max_model_len": 4096,
                            "enforce_eager": False,
                            "max_num_seqs": 256},
            tensor_parallel_size=2,
            stop=["User:", "\n"*14]
            )

elif args.model == 'llama2':

    model_path = "/m2/Models/hub/openbuddy-llama2-70B-v13.2-AWQ" # LLAMA 70B param model

    llm = CustVLLM(model=model_path,
            trust_remote_code=True,
            max_new_tokens=512,
            temperature=0.15,
            top_p=0.6,
            # supported quantizations: AWQ, GPTQ (e.g., 4-bit)
            vllm_kwargs={"quantization": "awq",
                            "gpu_memory_utilization": 0.8,
                            "max_model_len": 3072,
                            "enforce_eager": True,
                            "max_num_seqs": 256},
            tensor_parallel_size=2,
            stop=["User:", "\n"*14]
            )

from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import LLMChain

def return_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
                    session_id=session_id, \
                    connection_string="sqlite:///data/sqlite_backup.db"
                    )

@app.get("/")
def root_endpoint(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/v1/promptRequest")
async def promptRequest(request: Request) -> Response:

    request_dict = await request.json()

    input = request_dict.pop("input")

    output = llm.invoke(input)

    if verbose:
        logger.trace("Generated text:", output)

    output_dict = {"ai_msg": output}

    return JSONResponse(output_dict)

@app.post("/v1/chatRequest")
async def chatRequest(request: Request) -> Response:

    request_dict = await request.json()

    input = request_dict.pop("input")

    user_age = request_dict.pop("user_age")
    user_name = request_dict.pop("user_name")
    user_gender = request_dict.pop("user_gender")
    session_id = request_dict.pop("session_id")

    if verbose:
        logger.info("Session ID: %s" % session_id)
        logger.info("Incoming input: %s" % input)
        logger.trace("User's name: %s, user's age: %s" % (user_name, user_age))

    if "context" in request_dict:
        context = request_dict.pop("context")
    else:
        # qa_coef: tunable; the higher, the more important QA dataset would be
        context = RAG_inference(input, bump_qa = True, qa_coef = 1.15, \
                                verbose = verbose, \
                                logger = logger)

    if detect(input)['lang'] == 'en':
        user_language = 'English'
    elif detect(input)['lang'] == 'ru':
        user_language = 'Russian (на русском)'
    else:
        user_language = 'English'

    if user_gender == 'male':
        pronoun = 'he'
    else:
        pronoun = 'she'

    if user_gender not in ('male', 'female'):
        user_gender = "female"

    if int(user_age) < 35:
        few_shot = f"""User: Расскажи, что ты знаешь обо мне?
Klara AI: Я знаю только что, что ты указала в своём профиле :) Тебе {user_age} лет и тебя зовут {user_name}.
User: Окей, спасибо! Как думаешь, почему я об этом спросила?
Klara AI: Нет проблем :) Я думаю, ты спросила меня об этом, чтобы проверить, что мне о тебе известно, и действительно ли я могу тебе в чём-то помочь. И я надеюсь я действительно смогу тебе помочь. Зови меня Клара, я готова ответить на все твои вопросы!"""

    elif int(user_age) >= 35:
        few_shot = f"""User: Расскажи, что ты знаешь обо мне?
Klara AI: Я знаю только что, что вы указали в своём профиле :) Вам {user_age} лет и вас зовут {user_name}.
User: Окей, спасибо! Как думаешь, почему я об этом спросила?
Klara AI: Нет проблем! Я думаю, вы спросили меня об этом, чтобы проверить, что мне о вас известно, и действительно ли я могу вам в чём-то помочь. И я надеюсь я действительно смогу вам помочь. Зовите меня Клара, я готова ответить на все ваши вопросы!"""

    prompt = PromptTemplate(input_variables=['user_age',
                            'pronoun',
                            'few_shot',
                            'user_gender',
                            'user_name',
                            'user_language',
                            'context',
                            'input',
                            'history'], \
                            template = prompt_template)

    chat_history_obj = SQLChatMessageHistory(
                        session_id=session_id, \
                        connection_string="sqlite:///data/sqlite.db"
                        )

    memory = ConversationBufferWindowMemory(input_key='input', memory_key='history', \
                                        ai_prefix="Klara AI", human_prefix="User", \
                                        chat_memory=chat_history_obj, return_messages=False, \
                                        k = dialogue_window)

    llm_chain = LLMChain(prompt = prompt, llm = llm, memory = memory)

    chain_w_history = RunnableWithMessageHistory(
        llm_chain,
        get_session_history=return_session_history,
        input_messages_key="input",
        output_messages_key="text",
        history_messages_key="history"
    )

    output = chain_w_history.invoke(
        {"context": context,
        "input": input,
        "user_age": user_age,
        "user_name": user_name,
        "user_gender": user_gender,
        "pronoun": pronoun,
        "few_shot": few_shot,
        "user_language": user_language},
        config={"configurable": {"session_id": session_id}},
    )['text'].strip()

    if verbose:
        logger.success("Generated answer (session %s): %s" % (session_id, output))

    output_dict = {"ai_msg": output}

    return JSONResponse(output_dict)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)