import json
from datautils import utils
import nltk
from collections import Counter

import pickle
import numpy as np

QUESTION_CATEGORY_DICT = {'count':0,'exist':1,'query_color':2,'query_size':3,'query_actiontype':4,'query_direction':5,
        'query_shape':6,'compare_more':7,'compare_equal':8,'compare_less':9,'attribute_compare_color':10,'attribute_compare_size':11,
        'attribute_compare_actiontype':12,'attribute_compare_direction':13,'attribute_compare_shape':14}

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    with open(args.annotation_file, 'r') as anno_file:
        instances = json.load(anno_file)
    [video_ids.append(int(instance['id'])) for instance in instances]
    video_ids = set(video_ids)
    for video_id in video_ids:
        video_paths.append((args.video_dir + '{}.mp4'.format(str(video_id)), video_id))
    return video_paths


def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {} # train里的ans set
        for instance in instances:
            answer = instance['ans']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1} # 2 unknown-0 for train, 1 for val/test
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top) # top 4000-ans
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers) #都是answer出现的次数
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx) #把freq ans->load into dictionary
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question): # word+punctuation marks
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
# 将question的top freq vocab和ans的分开存放
    # Encode all questions
    print('Encoding data')
    questions_encoded = [] # question对应的index表示
    questions_len = []
    question_ids = []
    video_ids_tbw = []   # 存放的是对应的id
    video_names_tbw = [] # 同上-视频id
    all_answers = []
    question_category = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        im_name = int(instance['id'])
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)
        question_category.append(QUESTION_CATEGORY_DICT[instance['program'][-1]['function']])


        if instance['ans'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['ans']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)
    max_question_length = max(len(x) for x in questions_encoded) # max sequence length
    for qe in questions_encoded:
        while len(qe) < max_question_length: #每个都用0来padding
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32) # 真实句长
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()} # word+index
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
        'question_category':question_category
    }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)
