# Tiny GPT from scratch

A small character-level GPT trained on Tiny Shakespeare. It is the same core
architecture as GPT/ChatGPT: token embeddings, causal self-attention,
multi-head attention, residual transformer blocks, and a language-modeling
head. Everything is implemented directly in PyTorch with no pretrained model
and no transformer library.

Built from the ideas in Andrej Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and the
original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

## Results

Trained on CPU (Intel i5-12450H, PyTorch 2.13 CPU build):

| Model | Parameters | Initial val loss | Best val loss | Best step |
| --- | ---: | ---: | ---: | ---: |
| Bigram baseline | 4,225 | 4.7245 | 2.4822 | 750 |
| Character GPT | 816,512 (0.82M) | 4.1939 | **1.8868** | 3000 |

The GPT configuration is 4 layers, 4 attention heads, 128 embedding
dimensions, block size 128, dropout 0.1, batch size 32. The bigram model takes
about 10 seconds; the GPT takes about 45 minutes on CPU.

Resume line:

```text
Built and trained a 0.82M-parameter character-level transformer from scratch in PyTorch, reducing validation loss from 4.19 to 1.89 and generating coherent Shakespeare-style text.
```

### From noise to words

Step 0:

```text
qop--N'
ttmrgkMllWqt!KziJw?zFK:br:Mqs
lXRQA::WLPPvclMNE?TQi'DIXvVMBNwiqGi;EannpoGH-XEppTVkcZdF-HonxlXoI'hrdUBXaZUXOdT
```

Step 3000:

```text
UTER:
Yet.
There Cording the deack not of me come as gover
I hold  bid Lord Edwith hather the bide of your glo.

RORCANTIO:
Shall well to sooms the uch dother.
```

More samples are in `samples/`, including every 500 steps of the GPT run and
the final bigram baseline.

## Quick start

```powershell
python -m pip install -r requirements.txt
```

Tiny Shakespeare (~1.1 MB) is already at `data/input.txt`. If the file is
missing, the data loader downloads it automatically from the Karpathy
`char-rnn` repository.

Train the bigram baseline:

```powershell
python -m tinygpt.train --model bigram --max-steps 3000
```

Train the GPT:

```powershell
python -m tinygpt.train --model gpt --max-steps 3000
```

Generate from the best checkpoint:

```powershell
python -m tinygpt.sample --checkpoint runs/gpt.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 50
```

The checkpoint stores the character vocabulary, model configuration, step,
validation loss, and weights, so sampling does not need the training script.

## Browser demo

Start a local web interface and open it automatically:

```powershell
python -m tinygpt.web
```

Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000). The page lets you
edit the prompt, set temperature, top-k, token count, and seed, then generate
samples with the trained checkpoint. Everything runs locally on the CPU; no
request leaves the machine.

Use a different checkpoint or port:

```powershell
python -m tinygpt.web --checkpoint runs/bigram.pt --port 8080
```

Add `--no-open` to stop the browser from opening automatically.

## Temperature and top-k

`sample.py` accepts any temperature and top-k:

```powershell
python -m tinygpt.sample --temperature 0.5 --top-k 0
python -m tinygpt.sample --temperature 1.0 --top-k 10
python -m tinygpt.sample --temperature 1.2 --top-k 50
```

To compare settings side by side:

```powershell
python -m tinygpt.compare --checkpoint runs/gpt.pt --prompt "ROMEO:" --tokens 200 --temperatures 0.5,0.8,1.0,1.2 --top-k-values 0,10,50
```

This writes `samples/sampling_comparison.txt`. At temperature 0.5 the text is
repetitive but word-shaped. At 0.8 it produces the most convincing
Shakespeare-like dialogue. At 1.2 spelling and syntax become noticeably
looser. Top-k 10 removes the long tail of unlikely characters; top-k 50 is
close to unfiltered sampling for this model.

## Continue training

Resume from the best checkpoint and extend the cosine schedule:

```powershell
python -m tinygpt.train --resume-from runs/gpt.pt --max-steps 4000
```

`--max-steps` is the new total step count. Metrics are appended to
`runs/gpt_metrics.csv`, and the checkpoint architecture and vocabulary are
loaded from the checkpoint. Optimizer momentum is not restored, so the first
few resumed steps use fresh AdamW state.

## Use your own text

Put any UTF-8 text file anywhere and point `--data` at it:

```powershell
python -m tinygpt.train --model gpt --data path\to\lyrics.txt --run-name lyrics
python -m tinygpt.sample --checkpoint runs/lyrics.pt --prompt "Verse 1:" --tokens 300
```

The character vocabulary is rebuilt from the new file. For a small corpus,
reduce `--block-size` and possibly `--batch-size` so the validation split is
longer than the context window.

## How it works

- `tinygpt/data.py`: downloads the dataset, builds the character tokenizer,
  splits train/validation data, and samples random training windows.
- `tinygpt/model.py`: bigram baseline, causal single-head attention, multi-head
  attention, feed-forward block, pre-LayerNorm transformer block, and GPT.
- `tinygpt/train.py`: AdamW, warmup plus cosine learning-rate decay, gradient
  clipping, periodic validation, checkpointing, resume support, and samples.
- `tinygpt/sample.py`: loads a checkpoint and samples with temperature/top-k.
- `tinygpt/compare.py`: generates a grid of temperature/top-k comparisons.
- `tinygpt/web.py`: local browser UI and JSON generation API.
- `tinygpt/checkpoint.py`: shared model construction, device selection, and
  checkpoint save/load helpers.
- `tests/test_model.py`: tokenizer, shapes, loss, causal masking, generation,
  weight tying, and batch-alignment tests.

The GPT uses pre-normalization, GELU feed-forward layers, learned position
embeddings, residual connections, dropout, and weight tying between the token
embedding and output projection. The final parameter count includes the shared
embedding/output matrix once.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Off Campus

If you're using [Off Campus](https://www.offcampus.com) for student housing, you
can use my referral code:

```text
FKA6H5
```

## References

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
