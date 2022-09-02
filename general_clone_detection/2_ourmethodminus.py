# output:
# 14824 748 2140 8980 1100917
# precision
# 0.36368990650077304 0.017767307173182163 0.03524012859211269 0.07182998355787872 0.414145132395887
# true positive
# 0.9026724582198337
# recall
# 0.9997301672962763 0.9679144385026738 0.6710280373831776 0.3259465478841871 0.015329039337207074

import pickle as pkl
import numpy as np
from numpy.core.numeric import zeros_like, ones_like

def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[np.nonzero(l2 < 0)] = 0.0
    return np.sqrt(l2)

if __name__ == '__main__':
    [keys, emb] = pkl.load(open('./emb-minus.pkl','rb'))
    dist_mask_0 = zeros_like(np.ndarray((len(keys), len(keys)), dtype=int))
    dist_mask_1 = ones_like(np.ndarray((len(keys), len(keys)), dtype=int))

    dist1 = np.arange(len(keys)).repeat(len(keys)).reshape((len(keys), len(keys)))
    dist2 = dist1.transpose()
    dist_mask = np.where(dist1 < dist2, dist_mask_1, dist_mask_0)

    emb = np.asarray(emb)
    dist = l2_dist(emb, emb)
    clonepairs = np.nonzero(np.where((dist <= 0.6) & (dist_mask == 1), dist_mask_1, dist_mask_0))
    clonepairs = list(zip(clonepairs[0], clonepairs[1]))
    query = [(min(int(keys[i[0]]), int(keys[i[1]])),max(int(keys[i[0]]), int(keys[i[1]])))for i in clonepairs]
    groundtruth = pkl.load(open('../testingdata/groundtruth.pkl', 'rb'))
    
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

    for c in query:
        c = str(c)
        try:
            hit[keylist[clonedict[c]]] += 1
            pairs[keylist[clonedict[c]]].append(c)
        except KeyError:
            FP += 1

    print(GT12, GVST3, GST3, GMT3, GWT34)
    print('precision')
    print(hit['T12']/len(query), hit['VST3']/len(query), hit['ST3']/len(query), hit['MT3']/len(query),hit['WT34']/len(query))
    print('true positive')
    print(1-FP/len(query))
    print('recall')
    print(hit['T12']/GT12, hit['VST3']/GVST3, hit['ST3']/GST3, hit['MT3']/GMT3, hit['WT34']/GWT34)

