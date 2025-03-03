import functools
import os
import warnings
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ml_collections

import tensorflow as tf
import sentencepiece

from PIL import Image

from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding

import const
import Classifier as mn

print(jax.devices())
print('JAX backend:', jax.default_backend())

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

backend = jax.lib.xla_bridge.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")

# --------------------------------------------------------------

SEQLEN = 128
IMAGE_SIZE = 224
BATCH_SIZE = 8
TRAIN_IMAGE_SUBSYSTEM=True

MODEL_PATH = const.MODELS_PATH+'paligemma/paligemma.npz'
TOKENIZER_PATH = const.MODELS_PATH+'paligemma/pg_tokenizer.model'

model = None
tokenizer = None
params = None
decode = None
trainable_mask = None
data_sharding = None

train_dataset,val_dataset=None,None


# --------------------------------------------------------------


def create_food_safety_dataset(train_df=None,val_df=None,batch_size=8):
    global train_dataset,val_dataset,BATCH_SIZE

    BATCH_SIZE=batch_size

    if train_df is not None:
        train_paths,train_labels,train_questions=train_df['file_name'],train_df['gpt_reply'],train_df['question']

        train_dataset = tf.data.Dataset.from_tensor_slices((train_questions[:],train_paths[:], train_labels[:]))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_paths)).repeat()

    if val_df is not None:
        val_paths,val_labels,val_questions,aux_info=val_df['file_name'],val_df['gpt_reply'],val_df['question'],val_df['class']+':'+val_df['dcode']
        val_dataset = tf.data.Dataset.from_tensor_slices((val_questions,val_paths,val_labels,aux_info))


def val_data_iterator_custom():
    for example in val_dataset.as_numpy_iterator():
        prefix = example[0].decode().lower()
        imp = example[1].decode('utf-8')
        suffix = example[2].decode().lower()
        aux_info=example[3].decode()+':'+imp

        image = Image.open(imp)
        image = preprocess_image_val(image, size=IMAGE_SIZE)
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
        suffix_tokens, _, _, _ = preprocess_tokens(suffix, seqlen=SEQLEN)
        aux_info,_,_,_ = preprocess_tokens(aux_info, seqlen=SEQLEN)

        yield {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "mask_ar": np.asarray(mask_ar),
            "mask_input": np.asarray(mask_input),
            "original_text": np.asarray(suffix_tokens),
            "aux_info": np.asarray(aux_info)
        }


def train_data_iterator_custom():
    for example in train_dataset.as_numpy_iterator():
        prefix = example[0].decode().lower()
        imp = example[1].decode('utf-8')
        suffix = example[2].decode().lower()

        image = Image.open(imp)
        image = preprocess_image(image, size=IMAGE_SIZE)

        tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)

        yield {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "mask_ar": np.asarray(mask_ar),
            "mask_loss": np.asarray(mask_loss),
        }

# --------------------------------------------------------------


def prepare_model(model_name=None):
    global model, tokenizer, decode, data_sharding, params,trainable_mask,MODEL_PATH

    if model_name is not None:
        MODEL_PATH = os.path.join(const.MODELS_PATH,'paligemma', model_name+'.npz')

    model_config = ml_collections.FrozenConfigDict({
        "llm": {"vocab_size": 257_152},
        "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
    })
    model = paligemma.Model(**model_config)
    tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

    params = paligemma.load(None, MODEL_PATH, model_config)

    decode_fn = predict_fns.get_all(model)['decode']
    decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())
    print('done')

    def is_trainable_param(name, param):  # pylint: disable=unused-argument
        if name.startswith("llm/layers/attn/"):  return True
        if name.startswith("llm/"):              return False
        if name.startswith("img/"):              return TRAIN_IMAGE_SUBSYSTEM
        raise ValueError(f"Unexpected param name {name}")

    trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

    mesh = jax.sharding.Mesh(jax.devices(), ("data"))

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data"))

    params_sharding = big_vision.sharding.infer_sharding(
        params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

    warnings.filterwarnings(
        "ignore", message="Some donated buffers were not usable")

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
    def maybe_cast_to_f32(params, trainable):
        return jax.tree.map(lambda p, m: p.astype(jnp.float32) if m else p,
                            params, trainable)

    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_leaves = jax.tree.leaves(trainable_mask)
    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx] = maybe_cast_to_f32(params[idx], trainable)
        params[idx].block_until_ready()
    params = jax.tree.unflatten(treedef, params)

    def parameter_overview(params):
        for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
            print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

    print(" == Model params == ")
    parameter_overview(params)


# --------------------------------------------------------------


def preprocess_image(image, size=224):
    image = np.asarray(image)
    image=mn.augment(image)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    image = image[..., :3]
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0


def preprocess_tokens(prefix, suffix=None, seqlen=None):
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)
    mask_loss = [0] * len(tokens)

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)
        mask_loss += [1] * len(suffix)

    mask_input = [1] * len(tokens)
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))


def postprocess_tokens(tokens):
    tokens = tokens.tolist()
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)


# --------------------------------------------------------------


def preprocess_image_val(image, size=224):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]


# --------------------------------------------------------------


@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
    imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

    def loss_fn(params):
        text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
        logp = jax.nn.log_softmax(text_logits, axis=-1)

        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.

        return jnp.mean(example_loss)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Apply gradients to trainable params using SGD.
    def apply_grad(param, gradient, trainable):
        if not trainable: return param
        return param - learning_rate * gradient

    params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)

    return params, loss


def train_model(TRAIN_EXAMPLES = 512):
    global params

    LEARNING_RATE = 0.015

    TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE

    prepare_model()

    train_data_it=train_data_iterator_custom()

    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=TRAIN_STEPS + 1, base=LEARNING_RATE,
        decay_type="cosine", warmup_percent=0.10)

    t = time()

    for step in range(1, TRAIN_STEPS + 1):
        delta = time() - t
        t = time()
        examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        learning_rate = sched_fn(step)
        params, loss = update_fn(params, batch, learning_rate)
        if np.isnan(loss):
            exit(1)

        loss = jax.device_get(loss)
        print(f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}. Time taken {delta:.2f}")
        if loss==np.nan:
            exit()

    flat, _ = big_vision.utils.tree_flatten_with_names(params)
    with open(const.MODELS_PATH+f"VLLM/v2_foodSafety_PaliGemma-3b-{IMAGE_SIZE}px_{TRAIN_EXAMPLES}ex_bf16.npz", "wb") as f:
        np.savez(f, **{k: v for k, v in flat})


def make_predictions(data_iterator, *, num_examples=None,
                     batch_size=4, seqlen=SEQLEN, sampler="greedy"):
    outputs = []
    while True:
        examples = []
        try:
            for _ in range(batch_size):
                examples.append(next(data_iterator))
                examples[-1]["_mask"] = np.array(True)  # Indicates true example.
        except StopIteration:
            if len(examples) == 0:
                return outputs

        # Not enough examples to complete a batch. Pad by repeating last example.
        while len(examples) % batch_size:
            examples.append(dict(examples[-1]))
            examples[-1]["_mask"] = np.array(False)  # Indicates padding example.

        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        tokens = decode({"params": params}, batch=batch,
                        max_decode_len=seqlen, sampler=sampler)

        tokens, mask = jax.device_get((tokens, batch["_mask"]))
        tokens = tokens[mask]  # remove padding examples.
        responses = [postprocess_tokens(t) for t in tokens]

        for example, response in zip(examples, responses):
            outputs.append((example["image"], response, postprocess_tokens(example["original_text"]),postprocess_tokens(example["aux_info"])))
            if num_examples and len(outputs) >= num_examples:
                return outputs


def food_safety_predictions(to_csv=None):
    from os.path import basename
    from time import time

    t=time()

    res=[]
    for image, response, original_text, aux_info in make_predictions(val_data_iterator_custom(), batch_size=16):
        res.append(tuple(aux_info.split(':'))+(response,))

    df=pd.DataFrame(res,columns=['class','dcode','file_name','reply'])
    df['file_name']=df['file_name'].apply(lambda x: basename(x).strip())
    print('Paligemma prediction. Time taken:', time()-t)
    if to_csv:
        df.to_csv(const.RESULTS_PATH+"LLM/paligemma_val.csv",index=None)
    return df


def clear_memory():
    import gc
    import torch
    global backend, model, params, tokenizer, decode, data_sharding

    del model
    del params
    del tokenizer
    del decode
    del data_sharding
    del backend

    exclude = ['gc', 'sys', 'globals', 'name', 'del', 'exit', 'quit', 'In', 'Out', '_', '__', 'get_ipython','exclude','torch']
    for name in dir():
        if name not in exclude and not name.startswith("__"):
            del globals()[name]

    gc.collect()
    jax.clear_backends()
    jax.clear_caches()
    torch.cuda.empty_cache()
    print()
