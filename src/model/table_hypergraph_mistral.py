import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import numpy as np
from torch_geometric.data import Batch
from transformers.trainer_pt_utils import LabelSmoother
from src.global_path import global_path

BOS = '<s>'
EOS = '</s>'

IGNORE_INDEX = LabelSmoother.ignore_index
fetaqa_queation_len = 512


class TableHypergraphMistral(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.dataset_name = args.dataset
        self.num_token = args.num_token

        print('Loading Mistral')
        kwargs = {
            # "max_memory": {0: '80GiB'},
            # "max_memory": {0: '32GiB', 1: '32GiB', 2: '32GiB', 3: '32GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        if args.llm_frozen == 'False' and args.llm_lora == 'False' and args.do_eval == 'False':
            kwargs["max_memory"] = {0: '80GiB', 1: '80GiB'}


        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing Mistral!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        elif args.llm_lora == 'False':
            print("Training Mistral with SFT!")
        else:
            print("Training Mistral with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading Mistral!')

        hypergraph_config = AutoConfig.from_pretrained(f'{global_path}/models/google-bert/bert-base-uncased', trust_remote_code=True)
        hypergraph_config.update({
            "pre_norm": False, 
            "activation_dropout":0.1, 
            "gated_proj": False,
            "num_hidden_layers": args.gnn_num_layers,
            "num_attention_heads": args.gnn_num_heads,
            "hidden_size": args.gnn_hidden_dim,
        })
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            hypergraph_config,
        ).to(self.model.device)
        self.word_emb_pro = nn.Linear(args.gnn_in_dim, args.gnn_hidden_dim).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 4096),
        ).to(self.model.device)

        if self.num_token > 1:
            self.token_split = nn.Linear(args.gnn_hidden_dim, self.num_token * args.gnn_hidden_dim).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        graphs.x_s = self.word_emb_pro(graphs.x_s)
        graphs.x_t = self.word_emb_pro(graphs.x_t)
        s_embeds, t_embeds = self.graph_encoder(graphs)
        # mean pooling
        s_batch = torch.arange(graphs._num_graphs).repeat_interleave(graphs._slice_dict['x_s'][1:] - graphs._slice_dict['x_s'][:-1]).to(self.model.device)
        s_embeds = scatter(s_embeds, s_batch, dim=0, reduce='mean')
        t_batch = torch.arange(graphs._num_graphs).repeat_interleave(graphs._slice_dict['x_t'][1:] - graphs._slice_dict['x_t'][:-1]).to(self.model.device)
        t_embeds = scatter(t_embeds, t_batch, dim=0, reduce='mean')
        g_embeds = torch.stack([s_embeds, t_embeds], dim=0)
        g_embeds = torch.mean(g_embeds, dim=0)

        return g_embeds

    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        # eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        if self.num_token > 1:
            graph_embeds = self.token_split(graph_embeds)
            graph_embeds = graph_embeds.view(graph_embeds.shape[0], self.num_token, -1)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            if self.dataset_name == 'fetaqa':
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i][:fetaqa_queation_len] + label_input_ids
            else:
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if self.num_token > 1:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], inputs_embeds], dim=0)
            else:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            # bos + one_graph_emb + description + question + eos_user + label + eos

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)
            
        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        # eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        if self.num_token > 1:
            graph_embeds = self.token_split(graph_embeds)
            graph_embeds = graph_embeds.view(graph_embeds.shape[0], self.num_token, -1)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            if self.dataset_name == 'fetaqa':
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i][:fetaqa_queation_len]
            else:
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i]
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if self.num_token > 1:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], inputs_embeds], dim=0)
            else:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def inference_cut(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        # eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        cut_index = []
        for i in range(batch_size):
            # Add bos & eos token
            if self.dataset_name == 'fetaqa':
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i][:fetaqa_queation_len]
            else:
                if len(descriptions.input_ids[i]) > self.max_txt_len:
                    continue
                input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i]
            cut_index.append(i)
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        # for i in range(batch_size):
        for i in range(len(batch_inputs_embeds)):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': [samples['id'][i] for i in cut_index],
                'pred': pred,
                'label': [samples['label'][i] for i in cut_index],
                'question': [samples['question'][i] for i in cut_index],
                'desc': [samples['desc'][i] for i in cut_index], }
    
    def _normalize_importance(self, word_importance):
        """
        Normalize importance values of words in a sentence using min-max scaling.

        Parameters:
        - word_importance (list): List of importance values for each word.

        Returns:
        - list: Normalized importance values for each word.
        """
        min_importance = np.min(word_importance)
        max_importance = np.max(word_importance)
        return (word_importance - min_importance) / (max_importance - min_importance)
    
    def visualize_input_token_importance(self, sample):
        """
        Visualize tokens importance in an input based on gradient information.
        """
        sample['graph'] = Batch.from_data_list([sample['graph']])
        question = self.tokenizer(sample["question"], add_special_tokens=False)
        description = self.tokenizer(sample["desc"], add_special_tokens=False)
        label = self.tokenizer(sample["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))

        # encode graphs
        graph_embed = self.encode_graphs(sample)
        graph_embed = self.projector(graph_embed)

        label_input_ids = label.input_ids[:self.max_new_tokens] + eos_tokens.input_ids
        if self.dataset_name == 'fetaqa':
            input_ids = description.input_ids[:self.max_txt_len] + question.input_ids[:fetaqa_queation_len] + label_input_ids
        else:
            input_ids = description.input_ids[:self.max_txt_len] + question.input_ids + label_input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        inputs_embeds = torch.cat([bos_embeds, graph_embed, inputs_embeds], dim=0)  #bos_embeds [4, 4096]
        # bos + one_graph_emb + description + question + eos_user + label + eos

        input_attns = [1] * inputs_embeds.shape[0]
        attention_mask = torch.tensor(input_attns).to(self.model.device)
        label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
        label_input_ids = torch.tensor(label_input_ids).to(self.model.device)

        inputs_embeds = inputs_embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        label_input_ids = label_input_ids.unsqueeze(0)
        inputs_embeds.requires_grad_()
        inputs_embeds.retain_grad()
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        outputs.loss.backward()
        grads = inputs_embeds.grad
        grads = grads.squeeze(0)
        token_importance = [grad.norm().item() for grad in grads]
        normalized_importance = self._normalize_importance(token_importance)

        return normalized_importance, 4

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param