# output:
# 14824 748 2140 8980 1100917
# precision
# 0.5577289168094423 0.024319436512469067 0.05753854940034266 0.05373120121835142 0.04954311821816105
# true positive
# 0.7428612221587665
# recall
# 0.7905423637344846 0.6831550802139037 0.5649532710280374 0.1257238307349666 0.0009455753703503534


from doctest import OutputChecker
import pickle as pkl
from tqdm import tqdm

if __name__ == '__main__':
    blockpath = './nil_codeblock'
    with open(blockpath, 'r')as fp:
        blines = fp.readlines()
    blockdict={}
    for bidx, bl in enumerate(blines):
        bname= bl.split('.java,')[0][46:]
        blockdict[str(bidx)] = bname
    evalpath = './nil_clonepairs'
    groundtruth = pkl.load(
        open('./testingdata/groundtruth.pkl', 'rb'))
    clonedict = groundtruth[0]
    GT12 = groundtruth[1]
    GVST3 = groundtruth[2]
    GST3 = groundtruth[3]
    GMT3 = groundtruth[4]
    GWT34 = groundtruth[5]
    
    keylist = {1:'T12', 2:'VST3',3:'ST3',4:'MT3',5:'WT34'}
    hit = {'T12':0, 'VST3':0,'ST3':0,'MT3':0,'WT34':0}
    pairs =  {'T12':[],'VST3':[],'ST3':[],'MT3':[],'WT34':[]}
    FP = 0
    with open(evalpath,'r')as fp:
        query = fp.readlines()
    for c in query:
        c = str('('+blockdict[c[:-1].split(',')[0]]+', '+ blockdict[c[:-1].split(',')[1]]+')')
        try:
            clonetype=clonedict[c]
            hit[keylist[clonetype] ]+= 1
            pairs[keylist[clonetype]].append(c)
        except KeyError:
            FP += 1

    print(GT12, GVST3, GST3, GMT3, GWT34)
    print('precision')
    print(hit['T12']/len(query), hit['VST3']/len(query), hit['ST3']/len(query), hit['MT3']/len(query),hit['WT34']/len(query))
    print('true positive')
    print(1-FP/len(query))
    print('recall')
    print(hit['T12']/GT12, hit['VST3']/GVST3, hit['ST3']/GST3, hit['MT3']/GMT3, hit['WT34']/GWT34)
    
