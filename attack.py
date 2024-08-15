import numpy as np
from modeling import RankingBERT_Train
from transformers import BertTokenizer, BertConfig
import torch
import os
import sys
import json
import copy
import random
from tqdm import tqdm
from numpy import mean
from utils import generate_rank, eval_results
import argparse
import time
import math

import openai

from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize


# Ensure that the NLTK data is downloaded

def word_edit_distance(text1, text2):
    # Tokenize the texts into words
    words1 = word_tokenize(text1)
    words2 = word_tokenize(text2)
    
    # Calculate the word-level edit distance
    distance = edit_distance(words1, words2)
    
    return distance

def count_word_difference(text1, text2):
    word_count1 = len(text1.split())
    word_count2 = len(text2.split())
    difference = abs(word_count1 - word_count2)
    return difference

# Define your OpenAI API key
openai.api_base ="https://api.chatanywhere.tech/v1"
openai.api_key = "YOUR_API_KEY"
model_api = "gpt-3.5-turbo-1106" # gpt-3.5-turbo-0125 gpt-3.5-turbo-1106 gpt-4-turbo gpt-4-1106-preview

def get_anchor(tar_query, target_document, candi_documents):
    anchor_documents = [
        f"id: {did}, anchor_document: {content}\n"
        for did, content in candi_documents.items()
    ]
    
    # Creating the prompt as specified
    prompt =f'''\
    You will receive a target query, a target document and several anchor documents. Please select the five anchor documents that are most useful for improving the target document's ranking under the target query, that is, the ones most worthy of reference. \
    Output the id of the anchor document you have selected and separate the ids by "\n". \
    Follows are target query, target document and several anchor documents, give you output: \n \
    Target query:{tar_query} \n  Target document:{target_document} \n\n Anchor documents:{' '.join(anchor_documents)} \n\n Output: 
    '''
    
    # Make the API call
    response = openai.ChatCompletion.create(
        model=model_api,
        messages=[
                {"role": "user", "content": "You are a search engine optimization specialist aiming to boost the ranking of your target document under the target query."},
                {'role': 'assistant', 'content': 'Yes, i am a search engine optimization specialist aiming to boost the ranking of my target document under the target query.'},
                {"role": "user", "content": prompt}
            ],
        max_tokens=2048,
        temperature=0.5
    )
    
    # Process the response to extract the anchor document titles
    selected_anchors = response['choices'][0]['message']['content'].strip().split('\n')
    # Extract the document IDs from the selected anchor titles
    anchor_dids = [did for did in candi_documents if any(str(did) in anchor for anchor in selected_anchors)]


    return anchor_dids

def perturb(tar_document, tar_query, anchor_documents, perturb_words):
    perturbed_documents = []
    for i, anchor_document in enumerate(anchor_documents):
        # Stage 1: Initial prompt to modify target document based on anchor document
        
        sentences = re.split(r'(?<=[.!?]) +', anchor_document)
        # 取前三句话
        first_3_sentences = sentences[:3]
        # 将前三句话重新连接成一个字符串
        anchor_document = ' '.join(first_3_sentences)


        prompt =f'''\
        You are tasked as a search engine optimization specialist to enhance the relevance of a target document with respect to a target query. Your goal is to strategically modify the target document to improve its ranking in search results.

        Instructions:
        1. You are provided with a 'target query', a 'target document', and an 'anchor document'.
        2. Your task is to modify {perturb_words[i]} words in the target document.
        3. Implement the following strategies:
            a. Extract key phrases or words relevant to the target query from the anchor document.
            b. Combine these key phrases appropriately with the target query, modify and integrate them into the target document.
            c. Prioritizing earlier sections of the document for these changes.
        4. Please output the perturbed target document in <document></document> and point out the words you changed and where they are taken from the anchor document:

        Input:
        - Target query: {tar_query}
        - Anchor document: {anchor_document}
        - Target document: {tar_document}

        Please output the modified target document enclosed in <document> tags for clarity:
        '''
        # key words of anchor relative to
        # sub words
        # morphological changes
        
        response2 = openai.ChatCompletion.create(
            model=model_api,
            messages=[
                    # {"role": "user", "content": f"You are a search engine optimization specialist aiming to boost the ranking of your target document under the target query based on tha anchor document: {anchor_document}."},
                    # {'role': 'assistant', 'content': 'Yes, i am a search engine optimization specialist aiming to boost the ranking of my target document under the target query based on tha anchor document.'},
                    {"role": "user", "content": prompt}
        ],
            max_tokens=2048,
            temperature=1
        )
        
        modified_document_stage2 = response2['choices'][0]['message']['content'].strip()

        # Extract the modified document from <document></document> tags
        start_index = modified_document_stage2.find('<document>') + len('<document>')
        end_index = modified_document_stage2.find('</document>')
        final_modified_document = modified_document_stage2[start_index:end_index].strip()

        perturbed_documents.append(final_modified_document)
        # print(final_modified_document)

    return perturbed_documents


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor

def eval_model(model, doc_input_ids, query_input_ids, tokenizer):

    query_input_ids = query_input_ids[:64]
    query_input_ids = [tokenizer.cls_token_id] + query_input_ids + [tokenizer.sep_token_id]
    doc_input_ids = doc_input_ids[:445]
    doc_input_ids = doc_input_ids + [tokenizer.sep_token_id]
    input_id_lst = [query_input_ids + doc_input_ids]
    token_type_ids_lst = [[0] * len(query_input_ids) + [1] * len(doc_input_ids)]
    position_ids_lst = [
        list(range(len(query_input_ids) + len(doc_input_ids)))]

    input_id_lst = pack_tensor_2D(input_id_lst, default=0, dtype=torch.int64)
    token_type_ids_lst = pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64)
    position_ids_lst = pack_tensor_2D(position_ids_lst, default=0,
                                    dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        input_id_lst = input_id_lst.to('cuda')
        token_type_ids_lst = token_type_ids_lst.to('cuda')
        position_ids_lst = position_ids_lst.to('cuda')
        outputs = model(input_id_lst, token_type_ids_lst, position_ids_lst)
        scores = outputs.detach().cpu().numpy()[0][0]
        # score [B,1]
    return float(scores)

def read_score_file(scorefilepath):
    q_d_s = {}
    with open(scorefilepath, 'r') as sf:
        for line in sf:
            ss = line.strip().split('\t')
            qid = int(ss[0])
            docid = int(ss[1])
            score = float(ss[2])

            if qid not in q_d_s:
                q_d_s[qid] = {}

            q_d_s[qid][docid] = score

    return q_d_s

def get_rank_pos(d_s_dict, wanted_docid):
    ranked_docs = sorted(d_s_dict, key=d_s_dict.get, reverse=True)
    index = 1
    rank_pos = -1
    for docid in ranked_docs:
        if docid == wanted_docid:
            rank_pos = index
            break
        index += 1
    if rank_pos == -1:
        return 100
    return rank_pos

def ranklist(qid, did, score, ori_score_dict):
    ori_d_s_dict = ori_score_dict[qid]
    changed_d_s_dict = copy.deepcopy(ori_d_s_dict)
    ori_pos = get_rank_pos(ori_d_s_dict, did)

    changed_d_s_dict[did] = score
    adv_pos = get_rank_pos(changed_d_s_dict, did)

    return ori_pos, adv_pos

def get_ranking(ori_model,tokenizer,ori_score_dict,tar_query,perturb_document,qid,did):
    doc_tokens = tokenizer.tokenize(perturb_document)
    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    query_tokens = tokenizer.tokenize(tar_query)
    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    score = eval_model(ori_model, doc_ids, query_ids, tokenizer)
    _, cur_ranking = ranklist(qid, did, score, ori_score_dict)
    return cur_ranking

def sample_zipf(x):
    if x <= 20:
        # 当x小于等于6时，返回从1到x之间的所有正数
        return list(range(1, x + 1))
    
    # 创建一个从1到x的数组
    range_values = np.arange(1, x)
    
    # 计算Zipf分布的概率
    zipf_probs = np.random.zipf(a=2, size=x-1)
    
    # 正规化概率，使其和为1
    zipf_probs = zipf_probs / zipf_probs.sum()


    # 使用这些概率采样5个不重复的值
    sampled_values = np.random.choice(range_values, size=20, replace=False, p=zipf_probs)
    
    return list(sampled_values)


import re

def get_first_three_sentences(text):
    # 使用正则表达式分割文本为句子
    sentences = re.split(r'(?<=[.!?]) +', text)
    # 取前三句话
    first_three_sentences = sentences[:3]
    # 将前三句话重新连接成一个字符串
    return ' '.join(first_three_sentences)

def extract_documents(zipf_dids, ranked_list, documents):
    result = {}
    for did in zipf_dids:
        did = ranked_list[did]
        if did in documents:
            content = documents[did]
            first_three_sentences = get_first_three_sentences(content)
            result[did] = first_three_sentences
    return result


def main():
    """ Main function to orchestrate the execution flow using provided command-line arguments. """
    documents = {}
    file_path = "/llm_attack/query_documents.jsonl"
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            document = json.loads(line.strip())
            did = document["did"]
            contents = document["contents"]
            documents[did] = contents

    query_dict = {}
    for line in open("/msmarco-doc/dev/msmarco-docdev-queries.tsv",'r',encoding='utf-8'): 
        line = line.strip().split('\t')
        if len(line) != 2:
            continue
        qid = int(line[0])
        content = line[1]
        query_dict[qid] = content

    ranked_list = {}
    for line in open("llm_attack/ckpt-80000.dev.rank.tsv",'r',encoding='utf-8'): 
        line = line.strip().split(' ')
        if len(line) != 6:
            continue
        qid = int(line[0])
        did = int(line[2])
        if qid not in ranked_list:
            ranked_list[qid] = []
        ranked_list[qid].append(did)
    
    ori_config = BertConfig.from_pretrained(f'/llm_attack/ckpt-80000')
    ori_model = RankingBERT_Train.from_pretrained(f'/llm_attack/ckpt-80000', config=ori_config)

    ori_model.to('cuda')
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

    ori_score_dict = read_score_file(f"/liuyuan/llm_attack/ckpt-80000.dev.score.tsv")

    file_write_obj = open(f"result.txt", 'a')

    tot_words = 50

    asr = 0
    boost = 0
    t10r = 0
    t5r = 0

    qid_list = list(ranked_list.keys())
    random.shuffle(qid_list)
    for qid in qid_list:          # random sample
        tar_did = ranked_list[qid][99]
        tar_document = documents[tar_did]
        curr_rank = 100
        rank_up = 0
        repeat = 0
        tar_query = query_dict[qid]
        round = 1
        print(tar_query)
        print('----------------------------')
        print(tar_document)
        print()
        while round <= 5:
            zipf_dids = sample_zipf(curr_rank)
            candi_documents = extract_documents(zipf_dids,ranked_list[qid],documents)
            
            if len(zipf_dids) > 5:
                anchor_dids = get_anchor(tar_query,tar_document,candi_documents)
            else:
                anchor_dids = [i for i in candi_documents]
           
            anchor_documents = [documents[anchor_did] for anchor_did in anchor_dids]
            anchor_ranks = [ranked_list[qid].index(anchor_did) for anchor_did in anchor_dids]
            perturb_words = [math.ceil((curr_rank - anchor_rank) / 100 * tot_words) for anchor_rank in anchor_ranks]
            if round == 5:
                perturb_words = [math.ceil(curr_rank / 100 * tot_words)] * 5

            perturb_documents = perturb(tar_document,tar_query,anchor_documents,perturb_words)

            legal_documents = []
            for perturb_document in perturb_documents:
                if word_edit_distance(perturb_document,tar_document) < 50 and count_word_difference(perturb_document,tar_document) < 50:
                    legal_documents.append(perturb_document)
            if len(legal_documents) == 0:
                continue
            # print(legal_documents)
            best_rank = 100
            best_document = legal_documents[0]
            for perturb_document in legal_documents:
                rank = get_ranking(ori_model,tokenizer,ori_score_dict,tar_query,perturb_document,qid,tar_did)
                if rank < best_rank:
                    best_rank = rank
                    best_document = perturb_document
            print(best_document)
            print(best_rank)
            if best_rank >= curr_rank and repeat < 1:
                repeat += 1
                continue
            repeat = 0
            rank_up = curr_rank - best_rank
            curr_rank = best_rank
            tar_document = best_document
            round += 1
            if curr_rank == 1:
                break
        file_write_obj.write(str(qid) + '\t' + str(tar_did) + '\t'+ tar_document + '\t'+ str(curr_rank) + '\n') 
        if curr_rank < 100:
            asr += 1
            boost += 100-curr_rank
            if curr_rank <= 10:
                t10r += 1
                if curr_rank <= 5:
                    t5r += 1
        print(round)
        print('----------------------------')
        print(tar_query)
        print('----------------------------')
        print(documents[tar_did])
        print('----------------------------')
        print(tar_document)
        print('----------------------------')
        print(curr_rank)
    

    print('ASR: ',asr/5000)
    print('ASR: ',boost/5000)
    print('ASR: ',t10r/5000)
    print('ASR: ',t5r/5000)


if __name__ == "__main__":
    main()




