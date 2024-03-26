## Klara AI bot

### Architecture

<img src="./data/KlaraAI bot architecture.svg">

### Installation

Configured to work on a setup with 2 GPUs with 24 VRAM each (2x3090 or 2x4090).

Install conda env:
```
conda create --name klara python=3.10
```
Restart terminal & activate env:
```
conda activate klara
```

Install dependencies:
```
pip install -r requirements.txt
```

Launch:
```
python main.py
```
Or, to write all stdout (not just the logs) to a file:
```
python main.py > extended_logs.log
```
When running first time, one needs to specify path to model weights:
```
python main.py --model llama2 --model_path TheBloke/openbuddy-llama2-70B-v13.2-AWQ
```
or
```
python main.py --model mixtral --model_path TheBloke/openbuddy-mixtral-8x7b-v15.2-AWQ
```
Keyword args:
```
--model (optional: llama2 or mixtral; default: llama2)
--model_path (optional: TheBloke/openbuddy-llama2-70B-v13.2-AWQ or TheBloke/openbuddy-mixtral-8x7b-v15.2-AWQ)
--log_to_file (optional: true to store main logs in app.log; default: false)
--window (for dialogue's sliding window dialogue memory; default: 15)
```

#### AI Prompt Request

This endpoint accepts a prompt template as input and returns a response from the AI model.

*   Endpoint: `/v1/promptRequest`
*   Method: `POST`
*   Input: `input` (any text)
*   Output: `ai_msg` (response from AI model)

Example curl request:

```
curl -X POST -H "Content-Type: application/json" -d '{"input": "Your custom prompt here"}' http://localhost:8000/v1/promptRequest
```

#### AI Chat Request

This endpoint accepts a prompt template, context, user age, user gender, and session ID as input and returns a response from the AI model.

*   Endpoint: `/v1/chatRequest`
*   Method: `POST`
*   Input:  
    `input` (prompt template),  
    `context` (_optional_ context; if not provided, will be inferred by the Klara AI's retrieval model from files in data/),  
    `user_name` (user name),  
    `user_age` (user age),  
    `user_gender` (user gender),  
    `session_id` (session ID)
*   Output: `ai_msg` (response from the AI bot)

Example curl request:

```
curl -X POST -H "Content-Type: application/json" -d '{"input": "What services can Klara provide?", "context": "", "user_name": "Мария", "user_age": "27", "user_gender": "female", "session_id": "mariya_01f0"}' http://localhost:5001/v1/chatRequest
```
Example Python request:
```
input = "Привет, Клара! Что ты умеешь?"
user_age = "27"
user_name = "Мария"
user_gender = "female"
session_id = "mariya_01f0"

url = 'http://localhost:5001/v1/chatRequest'
request_dict = {
"input": input,
"user_age": user_age,
"user_name": user_name,
"user_gender": user_gender,
"session_id": session_id
}

response = requests.post(url, json=request_dict)
ai_msg = response.json()['ai_msg']
print("\nInput: %s" % input)
print("\nOutput: %s" % ai_msg)
```
