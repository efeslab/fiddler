import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers


def test():
    a = torch.range(1, 16)
    print(a)
    a = a.view(-1, 2, 2, 2)
    print(a.shape)
    print(a)
    a = a.view(-1, 8)
    print(a.shape)
    print(a)


def test1():
    # routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    selected_experts = torch.tensor([[0, 1], [0, 2]])
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=8)
    print(expert_mask, expert_mask.shape)
    expert_mask = expert_mask.permute(2, 1, 0)
    print(expert_mask, expert_mask.shape)


def test_softmax():
    a = torch.tensor([[1, 2, 3, 4], [1, 1, 1, 1]], dtype=torch.float32)
    a = F.softmax(a, dim=0)
    print(a)


def test_where():

    # Example binary mask (you can replace this with your actual data)
    expert_mask = torch.tensor([[[1, 0], [2, 1]], [[1, 2], [0, 2]]])

    # Assume i_expert is an index (e.g., i_expert = 0)
    i_expert = 0

    # Find indices where expert_mask[i_expert] is True
    idx, top_2 = torch.where(expert_mask[i_expert])
    print("Indices:", idx)
    print("Values at indices:", top_2)


def test_index():
    inputs = torch.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]], [[7, 8], [9, 10]]])
    inputs = inputs[[1, 2], [1, 1]]
    # inputs = inputs[[0, 1]]
    print(inputs)
    inputs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    inputs = inputs.reshape(-1, 4)
    print(inputs)


def test_shape():
    # inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    inputs = torch.tensor([[1], [2], [3], [4]])
    print(inputs.shape)
    inputs = inputs[:, :2]
    print(inputs)


def test_index_add():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    index = torch.tensor([0, 1])
    c = b.index_add(0, index, a)
    print(c)


def test_unsqueeze():
    output = torch.tensor([[1, 2], [3, 4]])
    print(output.shape)
    input_ids = output[:, -1].unsqueeze(-1)
    print(input_ids)


def test_argmax():
    a = torch.tensor(
        [[[1, 2, 3], [3, 4, 0], [2, 6, 9]], [[9, 6, 2], [7, 8, 3], [0, 4, 2]]]
    )
    print(a.shape)
    b = torch.argmax(a, dim=-1)
    print(b)
    print(b.shape)
    c = b[:, -1]
    print(c)
    c = c.unsqueeze(-1)
    print(c)
    print(c.shape)


def test_arrange():
    len_ = 1024
    position_ids = torch.arange(
        len_,
        len_ + 1,
        dtype=torch.long,
    )
    print(position_ids)
    position_ids = position_ids.unsqueeze(0).view(-1, 1)
    print(position_ids)


def test_topk():
    logits = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.3, 0.1]],
            [[0.4, 0.3, 0.2, 0.1], [0.2, 0.4, 0.1, 0.45]],
        ]
    )
    # logits = torch.tensor(
    #     [
    #         [[0.1, 0.2, 0.3, 0.4]],
    #         [[0.4, 0.3, 0.2, 0.1]],
    #     ]
    # )
    print(logits.shape)
    values, output = torch.topk(logits, 3, dim=-1)
    print(output)
    # output = output.view(-1, 1)
    # print(output)
    # output = output.view(-1, 1)
    # print(output)
    input_ids = output[:, -1]
    print(input_ids)
    col_idx = torch.tensor([[0], [1]])
    row_idx = torch.tensor([0, 1])
    input_ids = input_ids[row_idx, col_idx].view(-1, 1)
    print(input_ids)
    # values = values[:, -1]
    # values = values.flatten().view(-1, 1)
    # new_values = torch.full((values.shape[0], 1), 1.0)
    # new_values = new_values * values
    # print(new_values)
    # print(values)
    # print(input_ids)
    # print(input_ids.shape)
    # input_ids = input_ids.flatten().view(-1, 1)
    # print(input_ids)


def test_cat():
    a = torch.tensor([])
    b = torch.tensor([13, 14, 15])
    c = torch.cat((a, b), dim=0)
    print(c)
    c = torch.cat((c, b), dim=0)
    print(c)

    x = torch.arange(1, 2, dtype=torch.long)
    y = torch.tensor([0] * 5, dtype=torch.long).unsqueeze(0).view(-1, 1)
    print(y)


if __name__ == "__main__":
    test_topk()
    # test_cat()
    # test_unsqueeze()
    # test_argmax()
