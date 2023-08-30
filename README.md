### LocalLLM

There are several ways of running Large Language Models now a days. The very first options that comes in people's mind 
is just to use Open AI's API. Well it is all good. But the cost is also huge at the same time. Now when it comes to 
running LLMs in local, we:

- get confuse on which framework to choose to be our client (example: GPT4All, CTransformers, Llama.cpp etc)
- Managing CPU/GPU (i.e. resources) are a problem
- serving that LLM with the same API interface is also important but an overhead
- doing quick testing of the LLMs and comparing results with gpt APIs are also required before making to prod
- And there are lots when it comes to using and managing Local LLMs effectively. 


### Introducing LocalLLM

LocalLLM comes into rescue to solve all of this problems. Get started by just typing this command:

```bash
pip install localllm 
```

Our current supporting frameworks are:

- llama cpp python
- ctransformers
- gpt4all 
- huggingface

To install framework specific version of LocalLLM, you can just type:

```bash
pip install locallm['huggingface']
```

### Features we are trying to provide:

- Easy conversion of LLMs from huggingface to pytorch lightning or vice versa
- Conversion of the above models to GGUF (formerly GGML) or GPTQ format 
- easily create docker images for serving this models
- version control of the models
- stress testing of models in terms of performance and speed

### Contributing

This project is fully targetted to be open source. Contributions are welcome. Please reach out to [Anindyadeep](https://github.com/Anindyadeep) or feel free to raise issue to contribute on the project.
