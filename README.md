# LLaRA

##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/ljy0ustc/LLaRA.git
   cd LLaRA
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).

3. Download the data and checkpoints from https://drive.google.com/drive/folders/16I295mTyjnj97QVJud4LbCQ2lcGHqmY4?usp=drive_link.

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/` and the checkpoints to the dir path `checkpoints/`.

##### Train LLaRA

Train LLaRA with a single A100 GPU on the MovieLens dataset:

```sh
sh train_movielens.sh
```

Train LLaRA with a single A100 GPU on Steam dataset:

```sh
sh train_steam.sh
```

Note that: set the `llm_path` argument with your own directory path of the LLaMA2 model.

##### Evaluate LLaRA

Test LLaRA with a single A100 GPU on the MovieLens dataset:

```sh
sh test_movielens.sh
```

Test LLaRA with a single A100 GPU on Steam dataset:

```sh
sh test_steam.sh
```