import sys
import time
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="LDA evaluation")
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--count', required=True, help='Sampling count')
parser.add_argument('--vector', required=True, help='The number of gaussian vector')
parser.add_argument('--resize', required=True, help='Resize parameter')
parser.add_argument('--ratio', required=True, help='Sampling ratio')
args = parser.parse_args()

dataset = args.dataset
count = int(args.count)
vector = int(args.vector)
resize = int(args.resize)
ratio = float(args.ratio)

lda = pd.read_csv('./%s_resize%d_ratio%.6f_count%d_gvn%d_lda_log.txt'%(dataset, resize, ratio,count,vector), sep='\t', names=['timestamp', 'value'])
sb = pd.read_csv('./%s_resize%d_ratio%.6f_count%d_gvn%d_SB_log.txt'%(dataset, resize, ratio,count,vector), sep='\t', names=['timestamp', 'value'])
sw = pd.read_csv('./%s_resize%d_ratio%.6f_count%d_gvn%d_SW_log.txt'%(dataset, resize, ratio,count,vector), sep='\t', names=['timestamp', 'value'])

min_len = min(lda.shape[0], sb.shape[0], sw.shape[0])
print('Total results size : %d' % min_len, end='\n\n')
end_idx = min_len

print('Max lda =',lda['value'].max())
print('Max SB =',sb['value'].max())
print('Min, Max SW = (%.6f, %.6f)' % (sw['value'].min(), sw['value'].max()), end='\n\n')

print('Mean SB =',sb['value'].mean())
print('Mean SW =',sw['value'].mean())
print('Mean lda =',lda['value'][:end_idx].mean(), end='\n\n')

print('Var SB =',sb['value'].var())
print('Var SW =',sw['value'].var())
print('Var lda =',lda['value'].var(), end='\n\n')

t_tuple1 = time.strptime(sb['timestamp'][0][:-2].split('.')[0], '%Y-%m-%d %H:%M:%S')
t_tuple2 = time.strptime(sb['timestamp'][min_len-1][:-2].split('.')[0], '%Y-%m-%d %H:%M:%S')
time1 = time.mktime(t_tuple1)
time2 = time.mktime(t_tuple2)

ctime = time2 - time1
print('Elapsed time = %d hour %d min %d sec' % (ctime // 3600, (ctime % 3600) // 60, ctime % 60), end='\n\n')