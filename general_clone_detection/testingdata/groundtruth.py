import psycopg2
import pickle as pkl
from tqdm import tqdm
conn = psycopg2.connect(user="bcb", password="123",
                        database="bigclonebench")
cur = conn.cursor()

cur.execute('select (function_id_one, function_id_two, similarity_line, similarity_token) from clones where functionality_id >=25;')
clones = cur.fetchall()
clones = [eval(t[0]) for t in clones]

with open('./test.func.txt','r')as fp:
    keylist = fp.readlines()
keylist = [int(k[:-1]) for k in keylist]
T12 = 0
VST3 = 0
ST3 = 0
MT3 = 0
WT34 = 0

clonedict = {}

for i in tqdm(clones):
    if not i[0] in keylist:
        continue 
    if not i[1] in keylist:
        continue 
    simlarity = min(i[2],i[3])
    if simlarity == 1:
        clonedict[str((i[0],i[1]))] = 1
        T12 += 1
    elif simlarity >= 0.9:
        clonedict[str((i[0],i[1]))] = 2
        VST3 += 1
    elif simlarity >=0.7:
        clonedict[str((i[0],i[1]))] = 3
        ST3 += 1
    elif simlarity >= 0.5:
        clonedict[str((i[0],i[1]))] = 4
        MT3 += 1
    else:
        clonedict[str((i[0],i[1]))] = 5
        WT34 += 1
pkl.dump([clonedict,T12,VST3,ST3,MT3,WT34], open(
    './groundtruth.pkl', 'wb'))
print(T12,VST3,ST3,MT3,WT34)