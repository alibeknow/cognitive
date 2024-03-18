## Klara AI bot

### Architecture

![Alt text](./data/KlaraAI bot architecture.svg)
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
curl -X POST -H "Content-Type: application/json" -d '{"input": "Your prompt template here"}' http://localhost:8000/v1/promptRequest
```

#### AI Chat Request

This endpoint accepts a prompt template, context, user age, user gender, and session ID as input and returns a response from the AI model.

*   Endpoint: `/v1/chatRequest`
*   Method: `POST`
*   Input:  
    `input` (prompt template),  
    `context` (_optional_ context; if not provided, will be inferred by Klara AI'a retrieval model based on information in data/),  
    `user_name` (user name),  
    `user_age` (user age),  
    `user_gender` (user gender),  
    `session_id` (session ID)
*   Output: `ai_msg` (response from the AI bot)

Example curl request:

```
curl -X POST -H "Content-Type: application/json" -d '{"input": "What services can Klara provide?", "context": "", "user\_name": "Mariya", "user\_age": "27", "user\_gender": "female", "session\_id": "mariya\_01f0"}' http://localhost:8000/v1/chatRequest
```
