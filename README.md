# LLaRA

- *2024.7*: We have resolved several bugs within our code. Below are the most recent results of LLaRA.

|                | movielens  || steam    || lastfm   ||
|----------------|------------|------|----------|------|----------|------|
|                | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 |
| LLaRA(GRU4Rec) | 0.9684     | 0.4000 | 0.9840 | 0.4916 | 0.9672 | 0.4918 |
| LLaRA(Caser)   | 0.9684     | 0.4211 | 0.9519 | 0.4621 | 0.9754 | 0.4836 |
| LLaRA(SASRec)  | 0.9789     | 0.4526 | 0.9958 | 0.5051 | 0.9754 | 0.5246 |
- *2024.5*: We have updated the Steam dataset to a new version, in which we've addressed an issue that led to the repetition of certain data in the last interacted item of sequence.
- 🔥 *2024.3*: Our paper is accepted by SIGIR'24! Thank all Collaborators! 🎉🎉
- 🔥 *2024.3*: Our [datasets](https://huggingface.co/datasets/joyliao7777/LLaRA) and [checkpoints](https://huggingface.co/joyliao7777/LLaRA) are released on the huggingface.
  
##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/ljy0ustc/LLaRA.git
   cd LLaRA
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).

3. Download the data and checkpoints.

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/` and the checkpoints to the dir path `checkpoints/`.

##### Train LLaRA

Train LLaRA with a single A100 GPU on MovieLens dataset:

```sh
sh train_movielens.sh
```

Train LLaRA with a single A100 GPU on Steam dataset:

```sh
sh train_steam.sh
```

Train LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh train_lastfm.sh
```

Note that: set the `llm_path` argument with your own directory path of the Llama2 model.

##### Evaluate LLaRA

Test LLaRA with a single A100 GPU on MovieLens dataset:

```sh
sh test_movielens.sh
```

Test LLaRA with a single A100 GPU on Steam dataset:

```sh
sh test_steam.sh
```

Test LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh test_lastfm.sh
```