
#################### number of tokens ####################
# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 1


# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 2


# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 3


# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 5


# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 7


# mistral
CUDA_VISIBLE_DEVICES=0 python table_train.py \
--dataset wtq_orig \
--prompt_type mistral \
--model_name table_hypergraph_mistral \
--llm_model_name mistral_7b \
--seed 42 \
--project wtq_multitoken \
--max_txt_len 1024 \
--max_new_tokens 128 \
--output_dir .hyper-outputs-tokens/mistral-pt \
--gnn_model_name hyper \
--gnn_num_layers 1 \
--patience 3 \
--num_token 9