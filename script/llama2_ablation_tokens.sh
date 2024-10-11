
#################### number of tokens ####################
# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 1


# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 2


# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 3


# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 5


# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 7


# llama2
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type llama2 \
--model_name table_hypergraph_llm \
--llm_model_name 7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/llama2-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 9