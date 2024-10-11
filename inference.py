import os
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.config import parse_args_table_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from table_train import cal_robustness


def main(args):

    # Step 1: Set up seed
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)
    seed_everything(seed=seed)
    print(args)

    # Step 2: Build Node Classification Dataset
    test_dataset = load_dataset[args.dataset]('test', prompt_type=args.prompt_type)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    if args.second_dataset != '':
        second_test_dataset = load_dataset[args.second_dataset]('test', prompt_type=args.prompt_type)
        second_test_loader = DataLoader(second_test_dataset, batch_size=4, drop_last=False, pin_memory=False, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    args.max_txt_len = 4096
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](init_prompt=test_dataset.init_prompt, args=args)

    # # step 4. load from ck
    if os.path.exists(args.llm_ckpt_path):
        checkpoint = torch.load(args.llm_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print(f'load from ckpt {args.llm_ckpt_path}')

    # Step 5. Evaluating
    model.max_txt_len = 4096
    model.eval()
    # print(model)
    print('model.max_txt_len: ', model.max_txt_len)
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)
    
    # Step 6. Post-processing & Evaluating
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/bmix_model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    acc = eval_funcs[args.dataset](eval_output, path)
    print(f'{args.dataset} Test Acc/blue: {acc}')

    with open(f'{args.output_dir}/{args.dataset}/score.txt', 'a', encoding='utf-8') as file:
        file.write(f'{path} \nTest Acc/blue: {acc}\n')

    wandb.log({f'{args.dataset} Test Acc/blue': acc})

    # Step 7. Second Evaluating
    if args.second_dataset != '':
        model.eval()
        eval_output = []
        progress_bar_test = tqdm(range(len(second_test_loader)))
        for step, batch in enumerate(second_test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                eval_output.append(output)

            progress_bar_test.update(1)
    
        # with open('fetaqa+lora_eval.json', 'w') as file:
        #     json.dump(eval_output, file, indent=4)

        # Step 6. Post-processing & compute metrics  
        
        # with open('fetaqa+lora_eval.json', 'r') as file:
        #     eval_output = json.load(file)
        
        os.makedirs(f'{args.output_dir}/{args.second_dataset}', exist_ok=True)
        second_path = f'{args.output_dir}/{args.second_dataset}/bmix_model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_token{args.num_token}.csv'
        acc = eval_funcs[args.second_dataset](eval_output, second_path)
        print(f'{args.second_dataset} Test Acc/blue: {acc}')

        robust_str = cal_robustness(path, second_path)
        print(robust_str)
        with open(f'{args.output_dir}/{args.second_dataset}/score.txt', 'a', encoding='utf-8') as file:
            file.write(f'{second_path} \nTest Acc/blue: {acc}\n')
            file.write(f'\n{robust_str}\n')
        
        wandb.log({f'{args.second_dataset} Test Acc/blue': acc})


if __name__ == "__main__":

    args = parse_args_table_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
