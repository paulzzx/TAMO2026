from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.table_hypergraph_llm import TableHypergraphLLM

from src.model.mistral import Mistral
from src.model.table_hypergraph_mistral import TableHypergraphMistral
from src.model.pt_mistral import PromptTuningMistral

from src.global_path import global_path

load_model = {
    'inference_llm': LLM,

    # tablellama
    'llm': LLM,
    'pt_llm': PromptTuningLLM,
    'table_hypergraph_llm': TableHypergraphLLM,

    # mistral
    'mistral': Mistral,
    'table_hypergraph_mistral': TableHypergraphMistral,
    'pt_mistral': PromptTuningMistral,
}

# Replace the following with the model paths
llama_model_path = {
    '7b': f'{global_path}/models/meta-llama/Llama-2-7b-hf',
    '7b_chat': f'{global_path}/models/meta-llama/Llama-2-7b-chat-hf',
    'table_llama_7b': f'{global_path}/models/tablellama',
    'mistral_7b': f'{global_path}/models/mistralai/Mistral-7B-v0.1',
    'mistral_7b_instruct': f'{global_path}/models/mistralai/Mistral-7B-Instruct-v0.2',
}
