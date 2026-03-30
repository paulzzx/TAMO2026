<div align="center">
  <h2>TAMO: Table as a Modality for Large Language Models</h2>
</div>


## About our Work

To migrate the remarkable successes of Large Language Models (LLMs), the community has made numerous efforts to generalize them to the table reasoning tasks for the widely deployed tabular data. Despite that, in this work, by showing a probing experiment on our proposed StructQA benchmark, we postulate that even the most advanced LLMs (such as GPTs) may still fall short on coping with tabular data. More specifically, the current scheme often simply replies on serializing the tabular data, together with the meta information, then put them through the LLMs. We argue that the loss of the structural information and incomplete cell values persisted are the root of this shortcoming. In this work, we further propose **TAMO** (reimagine <u>**Ta**</u>ble representation <u>**a**</u>s <u>**a**</u>n independent <u>**Mo**</u>dality) that bears an ideology to treat the tables as an independent modality integrated with the text tokens. The resulted model in TAMO is a multimodal framework consisting of a hypergraph neural network as the global table encoder seamlessly integrated with the mainstream LLM. Empirical results on various benchmarking datasets, including HiTab, WikiTQ, WikiSQL, FeTaQA, and StructQA, have demonstrated significant improvement on generalization with an average relative gain by **42.65%**.

## Datasets

To evaluate the effectiveness of TAMO, we conduct extensive experiments on our proposed table structure understanding dataset **StructQA** and four public table reasoning benchmarks (HiTab, WikiTQ, WikiSQL, and FetaQA).  
The **StructQA** dataset is located in the "``dataset/structProbe``" folder. In the early stages of the experiment, we named it structProbe, signifying that this is a probe testing LLM's understanding of table structures. For details on its construction, please refer to the paper.

## How to Run

1. Install Requirements.  

```
conda create --name TAMO python=3.9
conda activate TAMO
bash requirements.sh
```

2. Download Datasets and Models  
   **Datasets:** The code for downloading the Hugging Face version of the four public table reasoning benchmark datasets (HiTab, WikiTQ, WikiSQL, and FetaQA) is located in the file "``download_dataset.py``".  
   **Models:** The script for downloading all the models used is contained within the "``download_models.sh``" file. It downloads both the primary runtime backbones and the auxiliary local assets used by the repo, and Llama 2 downloads require Hugging Face access approval plus `huggingface-cli login`.

3. Data Preprocess  
   Run the "``./script/data_preprocess.sh``" script to execute the data preprocess. This preprocessing flow remains single-GPU and uses `GPU0`.

4. Run Experiment  
   All the experimental run scripts are located in the "``./script``" folder. The intended train/inference hardware baseline is dual A800 80G with `GPU0` and `GPU1`, and the scripts are expected to run with `CUDA_VISIBLE_DEVICES=0,1`.
