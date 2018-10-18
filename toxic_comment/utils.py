import pandas as pd
import yaml
from attrdict import AttrDict


def load_data(pkl = True):    
    if pkl:
        train_data = pd.read_pickle('data/train_data.pkl')
        test_data = pd.read_pickle('data/test_data.pkl')
    else:
        train_data = pd.read_csv('data/train.csv')        
        test_data = pd.read_csv('data/test.csv')
    return(train_data, test_data)



def save_data(file_name, preds, toxic_types):
    submid = pd.read_csv('submission/sample_submission.csv')
    submid = pd.DataFrame({'id': submid["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = toxic_types)], axis=1)
    submission.to_csv('submission/' + file_name + '.csv', index=False)
    

def vocab_size(train_comment):
    a = set()
    for comment in train_comment:
        a.update(comment.split())
    return(len(a) + 1)
    
def read_yaml():
    with open('config.yaml') as f: config = yaml.load(f)
    return AttrDict(config)
        
    