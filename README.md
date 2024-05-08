
The repository for the paper 

**Analyzing Large Language Models for Classroom Discussion Assessment**. Nhat Tran, Benjamin Pierce, Diane Litman, Richard Corrent, Lindsay Clare Matsumura. **Educational Data Mining 2024**.
## Installation
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install "fschat[model_worker,webui]"
```
Download the vicuna-7b-16k model from [Huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k), rename it to `vicuna-7b-v1.5-16k` and put it to `models` folder (models/vicuna-7b-v1.5-16k)

Download the llama2 model from [Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main), rename it to `Llama-2-7b-hf` and put it to `models` folder (models/Llama-2-7b-hf)

## Experiments

Run `run_Vicuna.py` for the Vicuna models:

```
python run_Vicuna.py --model models/vicuna-7b-v1.5-16k --device gpu
```
Run `run_llama.py` for the Llama2 models:
```
python run_llama.py
```