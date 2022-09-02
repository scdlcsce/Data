# output:
# 14824 748 2140 8980 1100917
# precision
# 0.9100454377993369 0.036780056490237016 0.04777109173523272 0.0017806705145523762 0.0003070121576814442
# true positive
# 0.9966842686970404
# recall
# 0.9997976254722072 0.8008021390374331 0.36355140186915885 0.0032293986636971047 4.5416684454868075e-06


import pickle as pkl
from tqdm import tqdm

if __name__ == '__main__':
    blockpath = './nicad-test_functions-consistent-clones-0.40.xml'
    with open(blockpath, 'r')as fp:
        blines = fp.readlines()
    
    funclist = []
    for b in tqdm(blines):
        if '<source file=' in b:
            c = int(b.strip().split(' ')[1].split('/')[-1][:-6])
            funclist.append(c)

    clones = []
    for i in range(0,len(funclist),2):
        clones.append(str((funclist[i],funclist[i+1])) if funclist[i] < funclist[i+1] else str((funclist[i+1],funclist[i])))
    clones = list(set(clones))
 
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
    
    for c in clones:
        try:
            clonetype=clonedict[c]
            hit[keylist[clonetype] ]+= 1
            pairs[keylist[clonetype]].append(c)
        except KeyError:
            FP += 1

    print(GT12, GVST3, GST3, GMT3, GWT34)
    print('precision')
    print(hit['T12']/len(clones), hit['VST3']/len(clones), hit['ST3']/len(clones), hit['MT3']/len(clones),hit['WT34']/len(clones))
    print('true positive')
    print(1-FP/len(clones))
    print('recall')
    print(hit['T12']/GT12, hit['VST3']/GVST3, hit['ST3']/GST3, hit['MT3']/GMT3, hit['WT34']/GWT34)

