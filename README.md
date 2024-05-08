
The repository for the paper 

**Analyzing Large Language Models for Classroom Discussion Assessment**. Nhat Tran, Benjamin Pierce, Diane Litman, Richard Corrent, Lindsay Clare Matsumura. **Educational Data Mining 2024**.
## Installation
```
pip install -r requirements.txt
git lfs install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install "fschat[model_worker,webui]"
```

Create a folder named `models` to put the models in.
Create a folder named `data` where the transcripts are stored.

Download the Vicuna model from [Huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k), rename it to `vicuna-7b-v1.5-16k` and put it to `models` folder (models/vicuna-7b-v1.5-16k)

Download the Mistral model from [Huggingface](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), rename it to `
Mistral-7B-Instruct-v0.1 ` and put it to `models` folder (models/
Mistral-7B-Instruct-v0.1 )

## Experiments

Run `run_Vicuna.py` for the Vicuna models:

```
python run_Vicuna.py --model models/vicuna-7b-v1.5-16k --device gpu
```
Run `run_Mistral.py` for the Mistral models:
```
python run_Mistral.py
```