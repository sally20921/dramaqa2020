from contextlib import contextmanager
from datetime import datetime
from rotation import rotate_batch_with_labels
import os
import sys
import json
import pickle
import re

import six
import numpy as np
import torch

from config import log_keys

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(data, f, **kwargs)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]


def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


def make_jsonl(path, overwrite=False):
    if (overwrite or not path.is_file()) and path.suffix == '.jsonl':
        path_json = path.parent / path.name[:-1]
        with open(str(path_json), 'r') as f:
            li = json.load(f)
        with open(str(path), 'w') as f:
            for line in li:
                f.write("{}\n".format(json.dumps(line)))


def prepare_batch(args, batch, vocab):
    net_input_key = ['que', 'answers', *args.use_inputs]
    net_input = {k: batch[k] for k in net_input_key}
    batch_size = batch['que'].shape(0)

    image_input = batch['images']
    rot_90 = rotate_batch_with_labels(image_input, 1)
    rot_180 = rotate_batch_with_labels(image_input, 2)
    rot_270 = rotate_batch_with_labels(image_input, 3)
    rot_input = [rot_90, rot_180, rot_270]
    net_input_list = [net_input.copy() for _ in range(3)]
    for i in range(3):
        net_input_list[i]['image'] = rot_input[i]

    for i in range(3):
        for k in net_input_key:
            torch.cat([net_input['k'], net_input_list[i][k]], dim=1)

    for key, value in net_input.items():
        if torch.is_tensor(value):
            net_input[key] = value.to(args.device).contiguous()

    ans_idx = batch.get('correct_idx', None)
    if torch.is_tensor(ans_idx):
        ans_idx = ans_idx.to(args.device).contiguous()
    test_time_target = []
    for i in range(4):
        one_hot = [0,0,0,0]
        one_hot[i] = 1
        for j in range(batch_size):
            test_time_target.append(one_hot)
    test_time_target = torch.Tensor(test_time_target, dtype=torch.float32)
    return net_input, ans_idx, test_time_target


def to_string(vocab, x):
    # x: float tensor of size BSZ X LEN X VOCAB_SIZE
    # or idx tensor of size BSZ X LEN
    if x.dim() > 2:
        x = x.argmax(dim=-1)

    res = []
    for i in range(x.shape[0]):
        sent = x[i]
        li = []
        for j in sent:
            if j not in vocab.special_ids:
                li.append(vocab.itos[j])
        sent = ' '.join(li)
        res.append(sent)

    return res


def wait_for_key(key="y"):
    text = ""
    while (text != key):
        text = six.moves.input("Press {} to quit: ".format(key))
        if text == key:
            print("terminating process")
        else:
            print("key {} unrecognizable".format(key))


def get_max_size(t):
    if hasattr(t, 'shape'):
        if not torch.is_tensor(t):
            t = torch.from_numpy(t)
        return list(t.shape), t.dtype
    else:
        # get max
        t = [get_max_size(i) for i in t]
        dtype = t[0][1]
        t = [i[0] for i in t]
        return [len(t), *list(np.array(t).max(axis=0))], dtype


def pad_tensor(x, val=0):
    max_size, dtype = get_max_size(x)
    storage = torch.full(max_size, val).type(dtype)

    def add_data(ids, t):
        if hasattr(t, 'shape'):
            if not torch.is_tensor(t):
                t = torch.from_numpy(t)
            storage[tuple(ids)] = t
        else:
            for i in range(len(t)):
                add_data([*ids, i], t[i])

    add_data([], x)

    return storage


@contextmanager
def suppress_stdout(do=True):
    if do:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout



def get_episode_id(vid): 
    return int(vid[13:15]) # vid format: AnotherMissOh00_000_0000

def get_scene_id(vid):
    return int(vid[16:19]) # vid format: AnotherMissOh00_000_0000

def get_shot_id(vid):
    return int(vid[20:24]) # vid format: AnotherMissOh00_000_0000

frame_id_re = re.compile('IMAGE_(\d+)')
def get_frame_id(img_file_name):
    return int(frame_id_re.search(img_file_name).group(1)) # img_file_name format: IMAGE_0000070227

