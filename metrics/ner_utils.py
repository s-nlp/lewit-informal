import re

ones_C = [
         "zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"
         ]

ones_O = [
          "zero", "first", "second", "third", "fourth", "fifth", "sixth",
          "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
          "thirteenth", "fourteenth", "fifteenth", "sixteenth",
          "seventeenth", "eighteenth", "nineteenth"
          ]

tens_C = [
          "zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty",
          "seventy", "eighty", "ninety"
          ]

tens_O = [
          "zero", "tenth", "twentieth", "thirtieth", "fortieth",
          "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"
          ]
month_numbers = ['01', '1', '02', '2', '03', '3', '04', '4', '05', '5', '06',
                   '6', '07', '7', '08', '8', '09', '9', '10', '11', '12']
months = ['January', 'January', 'February', 'February', 'March',
                  'March', 'April', 'April', 'May', 'May', 'June', 'June',
                  'July', 'July', 'August', 'August', 'September', 'September',
                  'October', 'November', 'December']

mm = '(?P<month>' + '|'.join(list({m.lower() for m in months})) +')'
osmall = '((?P<tenth>' + '|'.join(tens_C + tens_O) + ')_)?(?P<oneth>' + '|'.join(ones_O + ones_C) + ')'

me1 = re.compile(mm + '_' + osmall)
me2 = re.compile(osmall + '_of_' + mm)

def norm_dates(text):
    for pattern in [me1, me2]:
        start_id = 0
        spans = []
        for match in re.finditer(pattern, text):
            gd = match.groupdict()
            if gd['oneth']:
                ones = ones_C.index(gd['oneth']) if gd['oneth'] in ones_C else ones_O.index(gd['oneth'])
            else:
                ones = 0
            if gd['tenth']:
                tens = tens_C.index(gd['tenth']) if gd['tenth'] in tens_C else tens_O.index(gd['tenth'])
            else:
                tens = 0
            normed = gd['month']
            if tens:
                normed += '_' + tens_C[tens]
            if ones:
                normed += '_' + ones_C[ones]
            first, last = match.span()
            spans.append(text[start_id:first])
            spans.append(normed)
            start_id = last
        spans.append(text[start_id:])
        text = ''.join(spans)
    return text



def adjust_money(token):
    token = re.sub("_\$_","_dollars_", token)
    return token

address_short2full = {'ave': 'avenue',
 'blvd': 'boulevard',
 'bldg': 'building',
 'crt': 'court',
 'cres': 'crescent',
 'dr': 'drive',
 'pl': 'place',
 'rd': 'road',
 'sq': 'square',
 'stn': 'station',
 'st': 'street',
 'terr': 'terrace'}

from nltk import wordpunct_tokenize

import sklearn
assert sklearn.__version__ == '0.22.1', 'https://github.com/EFord36/normalise/issues/124'

import normalise

# from normalise import tokenize_basic
from string import punctuation

def normalize_text_string(text, debug = False):
    text = re.sub(r'[^\w\s$]',' ',text)# remove noword nodigit
    text = re.sub(r'(?<=\d) (?=\d)','',text)#adjust phone numbers
    text = re.sub('(?<=\d)(th|rd|nd\st)', '' ,text)#adjust digits
    text = ' '.join([address_short2full[token] if token in address_short2full else token  for token in text.split()]) # expand addresses names
    if debug == True: print("after regex and address expansion", text)
#     text = re.sub(r'[\d]th ',' ',text)
#     print(text)
    try:
        tokens = normalise.normalise(text, tokenizer=wordpunct_tokenize, verbose=False, variety="AmE")
    except IndexError:
        tokens = text.lower().split()
    if debug == True: print("normalise", tokens)
    tokens_n = []
    for t in tokens:
        if len(t.split())>0:
            tokens_n.extend(t.split())
        else:
            tokens_n.extend(t)
    if debug == True:print(text)
    text = '_'.join([''] + tokens_n + ['']).lower().replace(':', '').replace('__', '_')
    if debug == True:print("tokens_n", text)
    text = re.sub('_[a](_.)?_? ?m(_.)?_', '_am_', text)
    text = re.sub('_[p](_.)?_? ?m(_.)?_', '_pm_', text)
    if debug == True:print(text)
    text = norm_dates(text)
    if debug == True:print(text)
#     text = adjust_dates(text)
#     if debug == True:print(text)
    text = adjust_money(text)
    return text


def get_hit_for_orig_token_and_gener_tokens_list(slot_value, rewritten_token_list):
    rewritten_token_list = [r.lower() for r in rewritten_token_list]
    hit = int(slot_value.lower() in rewritten_token_list)
    if not hit:
        nrs = normalize(slot_value)
        nrt = [normalize(t) for t in rewritten_token_list] #нормализовать по числам и по датам отдельно либо в одном строчке
        hit = nrs in nrt
    return hit


from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer 
lemmatizer_wnet = WordNetLemmatizer()
from nltk.corpus import wordnet
from tqdm import tqdm
import nltk
# nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

states_short2long = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming',
        'LA': 'Los Angeles',
    'NY': 'New York',
    'NYC': 'New York',
    'philly': 'Philadelphia',
    'SF' : 'San Francisco'
}

states_short2long = {k.lower():v.lower() for k,v in states_short2long.items()}
states_long2short = {v.lower():k.lower() for k,v in states_short2long.items()}


def adjust_state(dct , st):
    st_lower_str = st.lower().strip()
    st_lower_str_proc = re.sub('[^\w ]', '', st_lower_str)
    
    if st_lower_str_proc in dct.values():
        return st_lower_str_proc
    else:
        return dct.get(st_lower_str_proc)
    
def adjust_state_dot(dct , st):
    st_lower_str = st.lower().strip()
    st_lower_str_proc = re.sub('[^\w ]', '', st_lower_str)
    
    any_short_state =  '(' + '|'.join(list(dct.keys())) + ')'
    
    return re.sub(f'(?<=\s){any_short_state}(?!\w)', '' , st_lower_str_proc).strip()

from IPython.display import clear_output

def lemmatize_string(line):
        pos_tagged_ngramm = pos_tag(line.split())
        lemmatized_line_list = []
        for word_el in pos_tagged_ngramm:
            pos = get_wordnet_pos(word_el[1])
            if pos:
                lemma = lemmatizer_wnet.lemmatize(word_el[0], pos =pos)
            else:
                lemma = word_el[0]
            lemmatized_line_list.append(lemma)
        return ' '.join(lemmatized_line_list)

from IPython.display import clear_output

def get_ngrams(tokens, ngr_range = 4, connect_char = ' '):
    additional_ngrams = []
    # get more combinations of bi, tri, fur-grams
    for i, token in enumerate(tokens):
        collected_ngrams = [token]
        for j in range(1,ngr_range):
            target_index = i + j
            try:
                collected_ngrams.append(tokens[target_index])
                additional_ngrams.append(collected_ngrams.copy())
            except IndexError:
                break
      
    if connect_char == '':
        collected_ngrams_str = [[eli[:-1] for eli in el] for el in additional_ngrams]
        collected_ngrams_str = [f'{connect_char}'.join(el)+'_' for el in collected_ngrams_str]
    else:  
        collected_ngrams_str = [f'{connect_char}'.join(el) for el in additional_ngrams]
    
    tokens.extend(collected_ngrams_str)
  
import string
pucnt_dict = {}
for ch in string.punctuation:
    pucnt_dict[ch] = ' '
    
def adjust_dates(token):
    token = re.sub("in the (afternoon|evening|night)","pm", token)
    token = re.sub("in the morning","am", token)
    return token
   
def raw_string_preproc(raw_str):
    no_punct_raw_str = raw_str.translate(str.maketrans(pucnt_dict)).strip().lower()
    no_punct_raw_str = re.sub(' {2,10}', ' ', no_punct_raw_str )    
    
    no_punct_raw_str = re.sub("(?<=\d)[ -](?=\d)", '', no_punct_raw_str)# adjust phone numbers
    no_punct_raw_str = re.sub("\+(?=\d)", '', no_punct_raw_str)
    
    no_punct_raw_str = adjust_dates(no_punct_raw_str)
    
    return no_punct_raw_str

def get_necessary_ngram_size(slots_list):
    max_size = 0
    for sl in slots_list:
        cur_size = len(sl.split())
        max_size = max(max_size, cur_size)
        
    return max_size
    
from sacrebleu.metrics import CHRF 
import copy
chrf_clc = CHRF()
def get_ner_lists_smart_intersection(raw_generated_string, slot_list_orig, eps=1e-10, print_inersection = False):

    preproc_generated_str = raw_string_preproc(raw_generated_string)
    tokens_from_gener_string = preproc_generated_str.lower().split()
    slot_list_orig = [raw_string_preproc(el) for el in slot_list_orig]
    
#     print("\n preproc_generated_str", preproc_generated_str)
#     print("\n slot_list_orig", slot_list_orig)
    
    

    if len(tokens_from_gener_string) == 0 or len(slot_list_orig) == 0: 
        return [], 1

    # basic lemmatization
    lemm_gener_txt = lemmatize_string(preproc_generated_str) 
    lemm_slots = [lemmatize_string(n.lower().strip()) for n in slot_list_orig]
    
#     print("\nlemm_gener_txt",lemm_gener_txt)
#     print("lemm_slots",lemm_slots)
    
    # normalization
    normalized_gener_txt = normalize_text_string(preproc_generated_str, debug = False)
    normalized_slots = [normalize_text_string(n.lower().strip(), debug = False) for n in slot_list_orig]#slot_list_orig_lemm
    
#     print('\nnormalized_gener_txt',normalized_gener_txt)
#     print('normalized_slots',normalized_slots)
    
    # states
    states_orig_long = [adjust_state(states_short2long, n) for n in slot_list_orig ]
    states_orig_short = [adjust_state(states_long2short, n) for n in slot_list_orig ]
    states_orig_no_dot = [adjust_state_dot(states_short2long, n) for n in slot_list_orig ]

#     print("tokens_from_gener_string", tokens_from_gener_string)
    states_gener_long  = [adjust_state(states_short2long, n)  for n in tokens_from_gener_string]
    states_gener_short = [adjust_state(states_long2short, n)for n in tokens_from_gener_string]
    states_gener_no_dot = [adjust_state_dot(states_short2long, n) for n in tokens_from_gener_string ]
    
    collected_hits_tokens = []
    
    max_necess_ngram = get_necessary_ngram_size(slot_list_orig)
    max_necess_ngram = 3 if max_necess_ngram < 3 else max_necess_ngram
    if print_inersection == True: print(f"Will use window = {max_necess_ngram}")
    get_ngrams(tokens_from_gener_string, ngr_range = max_necess_ngram)
    ger_tokens_set = copy.copy(tokens_from_gener_string)
    ger_tokens_set = set(ger_tokens_set)

    ## CALCULATE DIRECT HITS
    for slot_orig_raw, slot_orig_lemm, slot_orig_norm, state_orig_l, state_orig_s, s_orig_no_dot  in zip(slot_list_orig, lemm_slots, 
                                                                       normalized_slots, 
                                                                        states_orig_long, states_orig_short, states_orig_no_dot):#elids_list_gener
        
        if print_inersection == True: 
            print("-"*100)
            print(slot_orig_raw) 
            
        if slot_orig_raw in preproc_generated_str:
            if print_inersection == True: print("slot_orig_raw {} found".format(slot_orig_lemm))
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':slot_orig_raw.lower(), 'score':1})
            
        elif slot_orig_lemm in lemm_gener_txt:
            if print_inersection == True: print("slot_orig_lemm {} found".format(slot_orig_lemm))
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':slot_orig_lemm.lower(), 'score':1})
        elif slot_orig_norm in normalized_gener_txt:
            if print_inersection == True: print("slot_orig_norm {} found".format(slot_orig_norm))
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':slot_orig_norm, 'score':1})  
          
        # STATES
        elif state_orig_l in states_gener_long and state_orig_l is not None:
            if print_inersection == True: print(f"state <{state_orig_l}> found")
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':state_orig_l, 'score':1}) 
                
        elif state_orig_s in states_gener_short and state_orig_s is not None:
            if print_inersection == True: print(f"state <{state_orig_s}> found")
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':state_orig_s, 'score':1})
            
        elif s_orig_no_dot.lower() in states_gener_no_dot:
            if print_inersection == True: print("slot_stateno dot {} found".format(slot_orig_lemm))
            collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found':s_orig_no_dot.lower(), 'score':1})
            
        # FALLBACK TO BAG OF TOKENS
        else:
            # calculate maximum tonek overlap in cimplicated and modificated slots
            slot_ngrams = slot_orig_raw.lower().split()
            get_ngrams(slot_ngrams)
                        
            slot_ngrams_set = set(slot_ngrams)
            
            if print_inersection == True: 
                print('slot_ngrams_set', slot_ngrams_set)
                print('ger_tokens_set', ger_tokens_set)
            
            inters_slot_tokens = slot_ngrams_set.intersection(ger_tokens_set)
            intersection_score = len(inters_slot_tokens) / len(slot_ngrams_set)
                
            if intersection_score > 0:
                if print_inersection == True: print("intersection ", intersection_score)
                collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found': inters_slot_tokens, 'score' : intersection_score})
            else:
                if print_inersection == True: print(f"Nothing found for {slot_orig_raw}, intersection_score = {intersection_score}")
                collected_hits_tokens.append({'raw_slot':slot_orig_raw, 'found': 'NOTHING_FOUND', 'score' : 0})
#     clear_output(wait=False)
    intersection_score_list = [el['score'] for el in collected_hits_tokens]  
    
    ## CALCULATE CHRF HITS 
    if print_inersection == True: print("\n CHRF Calculation...")
       
    chrf_intersection_score = []
    for slot_orig_raw, slot_orig_smart_score in zip(slot_list_orig, intersection_score_list):
        
        best_match_ngrm, best_crhf_score = None, None
        if slot_orig_smart_score != 1:
        
            tokens_count = len(slot_orig_raw.split())
            tokens_count = min(tokens_count, 4)

            ger_tokens_set_proper_ngram = [ngr for ngr in ger_tokens_set if len(ngr.split())==tokens_count]
            if len(ger_tokens_set_proper_ngram) == 0:
                ger_tokens_set_proper_ngram = copy.copy(ger_tokens_set)

            chrf_dict = {ref: chrf_clc.sentence_score(slot_orig_raw, [ref]).score/100 for ref in ger_tokens_set_proper_ngram}

            try:
                best_match_ngrm, best_crhf_score = sorted(chrf_dict.items(), key=lambda item: item[1], reverse = True)[0]
            except IndexError:            
                raise Exception ("STOP")

            chrf_intersection_score.append({'raw_slot':slot_orig_raw, 'found':best_match_ngrm, 'chrf_score':best_crhf_score})
        else:
            chrf_intersection_score.append({'raw_slot':slot_orig_raw, 'found':'FROM_SMART_SCORE', 'chrf_score':1})
        
        if print_inersection == True: 
            print("-"*100)
            print({'raw_slot':slot_orig_raw, 'found':best_match_ngrm if best_match_ngrm is not None else 'FROM_SMART_SCORE', 
                   'chrf_score':best_crhf_score if best_crhf_score is not None else 1})
    
    final_intersection_list = [max(sm['chrf_score'],chrf) for sm,chrf in zip (chrf_intersection_score, intersection_score_list)]
    final_intersection_score = sum(final_intersection_list) / (len(slot_list_orig) + eps)
    
    final_res = []
    for smart_token, chhrf_token, final_score in zip(collected_hits_tokens, chrf_intersection_score, final_intersection_list):
        report = {}
        report['raw_slot'] = smart_token['raw_slot']
        report['smart_found'] = smart_token['found']
        report['smart_score'] = smart_token['score']
        
        report['chrf_found'] = chhrf_token['found']
        report['chrf_score'] = chhrf_token['chrf_score']
        
        report['fin_score'] = final_score
        
        final_res.append(report)
            
    assert final_intersection_score <= 1
    
    return final_res, final_intersection_score
