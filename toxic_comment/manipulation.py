import re, string
from utils import read_yaml
from sklearn.model_selection import train_test_split


params = read_yaml()
toxic_types = params.toxic_types


def clean_word(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    if isinstance(s, str):
        return ' '.join(re_tok.sub(r' \1 ', s.lower()).split())
    else:
        return 'nan'

    
def clean_subword(s, n = 3):
    if isinstance(s, str):        
        s1 = re.sub('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’' + "']+",'', s.strip().lower())
        s2 = re.sub('[\s\t\n\r\b\a\f\v]+', '_', s1.strip())
        s3 = ' '.join([s2[i:i+n] for i in range(0, len(s2)- n + 1)])
        return(s3)
    else:
        return('nan')
        
        
def blending_data_split(train_comment, train_data, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(train_data['comment_text'], 
                                                        train_data[toxic_types], 
                                                        test_size= test_size, 
                                                        random_state = random_state)