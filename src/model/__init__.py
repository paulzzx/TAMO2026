from src.global_path import global_path


def _build_llm(*args, **kwargs):
    from src.model.llm import LLM
    return LLM(*args, **kwargs)


def _build_pt_llm(*args, **kwargs):
    from src.model.pt_llm import PromptTuningLLM
    return PromptTuningLLM(*args, **kwargs)


def _build_table_hypergraph_llm(*args, **kwargs):
    from src.model.table_hypergraph_llm import TableHypergraphLLM
    return TableHypergraphLLM(*args, **kwargs)


def _build_mistral(*args, **kwargs):
    from src.model.mistral import Mistral
    return Mistral(*args, **kwargs)


def _build_table_hypergraph_mistral(*args, **kwargs):
    from src.model.table_hypergraph_mistral import TableHypergraphMistral
    return TableHypergraphMistral(*args, **kwargs)


def _build_pt_mistral(*args, **kwargs):
    from src.model.pt_mistral import PromptTuningMistral
    return PromptTuningMistral(*args, **kwargs)

load_model = {
    'inference_llm': _build_llm,

    # tablellama
    'llm': _build_llm,
    'pt_llm': _build_pt_llm,
    'table_hypergraph_llm': _build_table_hypergraph_llm,

    # mistral
    'mistral': _build_mistral,
    'table_hypergraph_mistral': _build_table_hypergraph_mistral,
    'pt_mistral': _build_pt_mistral,
}

# Replace the following with the model paths
llama_model_path = {
    '7b': f'{global_path}/models/meta-llama/Llama-2-7b-hf',
    '7b_chat': f'{global_path}/models/meta-llama/Llama-2-7b-chat-hf',
    'table_llama_7b': f'{global_path}/models/tablellama',
    'mistral_7b': f'{global_path}/models/mistralai/Mistral-7B-v0.1',
    'mistral_7b_instruct': f'{global_path}/models/mistralai/Mistral-7B-Instruct-v0.2',
}
