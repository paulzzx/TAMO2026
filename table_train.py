import os
import wandb
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
import random

from src.model import load_model
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_table_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate

from src.global_path import global_path

import json
from collections import defaultdict


def validate_runtime_args(args):
    from src.model import ensure_known_llm_key, ensure_known_model_key, resolve_llm_model_path

    ensure_known_model_key(args.model_name)
    ensure_known_llm_key(args.llm_model_name)
    args.llm_model_path = resolve_llm_model_path(args.llm_model_name)
    validate_visible_gpus(args.expected_num_gpus)


def validate_visible_gpus(expected_num_gpus):
    if expected_num_gpus <= 1:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    requested = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    if len(requested) < expected_num_gpus:
        raise RuntimeError(
            f"This command expects at least {expected_num_gpus} visible GPUs, "
            f"but CUDA_VISIBLE_DEVICES='{visible or '<unset>'}'."
        )

    if torch.cuda.device_count() < expected_num_gpus:
        raise RuntimeError(
            f"This command expects at least {expected_num_gpus} visible CUDA devices, "
            f"but torch reports {torch.cuda.device_count()}. "
            f"Target baseline: dual A800 80G on GPU0 and GPU1."
        )


def cal_robustness(Origin_res_path, Permute_res_path):
    def evaluate_example(predict_str: str, ground_str: str, delimiter=", "):
        predict_str = predict_str.lower()
        ground_str = ground_str.lower()
        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            try:
                predict_values[float(span)] += 1
            except ValueError:
                predict_values[span.strip()] += 1
        for span in ground_spans:
            try:
                ground_values[float(span)] += 1
            except ValueError:
                ground_values[span.strip()] += 1
        _is_correct = predict_values == ground_values
        return _is_correct

    with open(Origin_res_path, 'r') as f:
        Origin_res = []
        for line in f:
            Origin_res.append(json.loads(line))
    Origin_right_index = []
    for i, res in enumerate(Origin_res):
        if evaluate_example(res['pred'], res['label']):
            Origin_right_index.append(i)

    with open(Permute_res_path, 'r') as f:
        Permute_res = []
        for line in f:
            Permute_res.append(json.loads(line))
    Permute_right_index = []
    for i, res in enumerate(Permute_res):
        if evaluate_example(res['pred'], res['label']):
            Permute_right_index.append(i)

    output_str = ''
    output_str += f"###################check answer###################" + '\n'
    output_str += f'Origin : {len(Origin_right_index)}, {len(Origin_right_index) / len(Origin_res)}' + '\n'
    output_str += f'Permute : {len(Permute_right_index)}, {len(Permute_right_index) / len(Permute_res)}' + '\n'
    output_str += f'ALL Sample : {len(Origin_res)}' + '\n'
    output_str += f"###################check answer###################" + '\n'
    output_str += f"###################calculate robustness###################" + '\n'
    same_count = 0
    for i, res in enumerate(Origin_res):
        if evaluate_example(Origin_res[i]['pred'], Permute_res[i]['pred']):
            same_count += 1
    output_str += f'Robustness : {same_count}, {same_count / len(Origin_res)}' + '\n'
    output_str += f"###################calculate robustness###################" + '\n'
    return output_str


def main(args):
    validate_runtime_args(args)

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               #    name=f"{args.dataset}_{args.model_name}_seed{seed}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}_numtoken{args.num_token}",
               config=args)

    seed_everything(seed=args.seed)
    print(args)

    # Step 2: Build Node Classification Dataset
    train_dataset = load_dataset[args.dataset]('train', prompt_type=args.prompt_type)
    val_dataset = load_dataset[args.dataset]('validation', prompt_type=args.prompt_type)
    test_dataset = load_dataset[args.dataset]('test', prompt_type=args.prompt_type)

    if args.second_dataset != '':
        second_test_dataset = load_dataset[args.second_dataset]('test', prompt_type=args.prompt_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=False, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=False, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=False,
                             shuffle=False, collate_fn=collate_fn)
    if args.second_dataset != '':
        second_test_loader = DataLoader(second_test_dataset, batch_size=4, drop_last=False, pin_memory=False,
                                        shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    model = load_model[args.model_name](init_prompt=train_dataset.init_prompt, args=args)

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training

    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_token{args.num_token}.csv'

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps), desc='Training', unit='step')
    best_val_loss = float('inf')
    best_test_acc = -float('inf')
    best_val_acc = -float('inf')

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            try:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()

                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

                if (step + 1) % args.grad_steps == 0:
                    adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

                optimizer.step()
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

                if (step + 1) % args.grad_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    wandb.log({'Lr': lr})
                    wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                    accum_loss = 0.

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Out of memory error encountered at batch {step}. Skipping this batch.")
                    torch.cuda.empty_cache()
                    progress_bar.update(1)
                    continue
                else:
                    raise

            progress_bar.set_postfix(loss=f'{loss.item():.4f}')
            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        val_loss = 0.
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        # print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')
        print(f'Epoch {epoch} Val Loss {val_loss}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating
    model = _reload_best_model(model, args)
    model.max_txt_len = 4096
    model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)

    # Step 6. Post-processing & compute metrics
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_token{args.num_token}.csv'
    acc = eval_funcs[args.dataset](eval_output, path)
    print(f'{args.dataset} Test Acc/blue: {acc}')

    with open(f'{args.output_dir}/{args.dataset}/score.txt', 'a', encoding='utf-8') as file:
        file.write(f'{path} \nTest Acc/blue: {acc}\n')

    wandb.log({f'{args.dataset} Test Acc/blue': acc})

    # Step 6. Second Evaluating
    if args.second_dataset != '':
        model = _reload_best_model(model, args)
        model.eval()
        eval_output = []
        progress_bar_test = tqdm(range(len(second_test_loader)))
        for step, batch in enumerate(second_test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                eval_output.append(output)

            progress_bar_test.update(1)
        os.makedirs(f'{args.output_dir}/{args.second_dataset}', exist_ok=True)
        second_path = f'{args.output_dir}/{args.second_dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_token{args.num_token}.csv'
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
