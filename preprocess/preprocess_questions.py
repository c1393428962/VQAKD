import argparse
import numpy as np
import os
import requests

from datautils import svqa
from datautils import msrvtt_qa
from datautils import msvd_qa

if __name__ == '__main__':
    # python preprocess/preprocess_questions.py --mode train
    # python preprocess/preprocess_questions.py --mode val
    # python preprocess/preprocess_questions.py --mode test
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='msvd-qa', choices=['msrvtt-qa', 'msvd-qa', 'svqa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode',
                        default='/root/autodl-tmp/feature-data/glove/glove.840.300d.pkl')
    parser.add_argument('--output_pt', type=str, default='/root/autodl-tmp/feature-data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='/root/autodl-tmp/feature-data/{}/{}_vocab.json')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'msrvtt-qa':
        args.annotation_file = '/home/WangJY/Jianyu_wang/MSRVTT-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/root/autodl-tmp/msvd-data/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('/root/autodl-tmp/feature-data/{}'.format(args.dataset)):
            os.makedirs('/root/autodl-tmp/feature-data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)
    elif args.dataset == 'svqa':
        args.annotation_file = '/home/WangJY/Jianyu_wang/SVQA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        svqa.process_questions(args)

    ### weixin token
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "69fcd3bf894d",
                             "title": "run status",
                             "name": "preprocess_question",
                             "content": "run success!!!"
                         })
    print(resp.content.decode())
