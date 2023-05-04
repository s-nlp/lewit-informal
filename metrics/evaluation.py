import argparse
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sacrebleu.metrics import BLEU, CHRF  # sacrebleu ~= 2.0.0

from transformers import logging
logging.set_verbosity_error()


def load_model(model_name=None, model=None, tokenizer=None, model_class=AutoModelForSequenceClassification, use_auth_token=None):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name, use_auth_token=use_auth_token)
        if torch.cuda.is_available():
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    return model, tokenizer


def prepare_target_label(model, target_label):
    if target_label is None:
        return 0
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(f'target_label "{target_label}" is not in model labels or ids: {model.config.id2label}.')
    return target_label


def classify_texts(texts, second_texts=None, model_name=None, target_label=None, batch_size=32, verbose=False, model=None, tokenizer=None, use_auth_token=None):
    model, tokenizer = load_model(model_name, model, tokenizer, use_auth_token=None)
    target_label = prepare_target_label(model, target_label)
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i:i+batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i:i+batch_size])
        inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            if logits.shape[1] == 1:  # single-label model
                preds = torch.sigmoid(logits)[:, 0].cpu().numpy()
            else:
                preds = torch.softmax(logits, -1)[:, target_label].cpu().numpy()
        res.append(preds)
    return np.concatenate(res)


def evaluate_formality(
    texts, 
    model='cointegrated/roberta-base-formality', 
    tokenizer = None,
    target_label=1,  # 1 is formal, 0 is informal
    batch_size=32, 
    verbose=False, 
    use_auth_token=None,
):
    if isinstance(model, str):
        model, tokenizer = load_model(model_name, model, tokenizer, use_auth_token=use_auth_token)
    else:
        assert tokenizer is not None
        
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        texts, 
        model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores


def mis_similarity(original_or_ref_text, generated_text, model):
    return model.compute(original_or_ref_text, generated_text)


def evaluate_meaning(
    original_texts, 
    rewritten_texts, 
    model=None,
    tokenizer=None,
    model_name='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', 
    target_label='entailment', 
    bidirectional=True, 
    batch_size=32, 
    verbose=False, 
    aggregation='prod',
    use_auth_token=None,
):
    if model_name == 'MIS':     
        assert model is not None
        return mis_similarity(original_texts, rewritten_texts, model)
    
    if model is None:
        model, tokenizer = load_model(model_name, model, tokenizer, use_auth_token=use_auth_token)      
            
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        original_texts, rewritten_texts, 
        model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    if bidirectional:
        reverse_scores = classify_texts(
            rewritten_texts, original_texts, 
            model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
        )
        if aggregation == 'prod':
            scores = reverse_scores * scores
        elif aggregation == 'mean':
            scores = (reverse_scores + scores) / 2
        elif aggregation == 'f1':
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores


def transpose_refs(refs):
    return list(zip(*refs))


def evaluate_meaning_ref(
    refs, 
    rewritten_texts, 
    model_name='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', 
    tokenizer = None,
    target_label='entailment',
    model=None,
    use_auth_token=None,
    **kwargs
):
    
    if model_name == 'MIS':     
        assert model is not None
        return mis_similarity(refs, rewritten_texts, model)
    
    if len(refs[0]) != len(rewritten_texts):
        refs = transpose_refs(refs)
    if len(refs[0]) != len(rewritten_texts):
        raise ValueError(f'Length of references ({len(refs[0])}) differs from length of rewritten texts ({len(rewritten_texts)})')

    model, tokenizer = load_model(model_name, model, tokenizer, use_auth_token=use_auth_token)
    target_label = prepare_target_label(model, target_label)
    
    all_ref_scores = np.stack([evaluate_meaning(r, rewritten_texts, model=model, tokenizer=tokenizer, target_label=target_label, **kwargs) for r in refs]).T
    best = all_ref_scores.max(1)
    return best


def evaluate_cola(
    texts, 
    model_name='textattack/roberta-base-CoLA', 
    target_label=1,
    batch_size=32, 
    verbose=False, 
    model=None,
    tokenizer=None,
    use_auth_token=None,
):
    model, tokenizer = load_model(model_name, model, tokenizer, use_auth_token=use_auth_token)
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        texts, 
        model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores


def get_text_perplexity(text, model, tokenizer, eos='\n', bos='\n'):
    encodings = tokenizer(eos + text + bos, return_tensors='pt', truncation=True)
    input_ids = encodings.input_ids.to(model.device)
    n_tokens = max(0, len(input_ids[0]) - 1)
    if n_tokens > 0:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
    else:
        loss = 0
    return loss, n_tokens

def get_corpus_perplexity(texts, model, tokenizer, unit='token', verbose=True, **kwargs):
    loss = []
    n_tokens = []
    pb = tqdm(texts) if verbose else  texts
    for text in pb:
        ll, w = get_text_perplexity(text, model, tokenizer, **kwargs)
        loss.append(ll)
        n_tokens.append(w)
    loss = np.array(loss)
    n_tokens = np.array(n_tokens)
    if unit == 'token': 
        return loss
    elif unit == 'text':
        return loss * n_tokens
    elif unit == 'char':
        return loss * n_tokens / [len(t) for t in texts]
    else:
        raise ValueError('unit should be one of ["token", "text", "char"]')


def evaluate_perplexity(
    texts, 
    original_texts=None,
    refs=None,
    model_name='gpt2-medium',
    unit='token',
    verbose=False, 
    model=None,
    tokenizer=None,
    comparison='relative',
    use_auth_token=None,
):
    model, tokenizer = load_model(model_name, model=model, tokenizer=tokenizer, model_class=AutoModelForCausalLM, use_auth_token=use_auth_token)
    scores = -get_corpus_perplexity(texts, model, tokenizer, unit=unit, verbose=verbose)
    if comparison and original_texts is not None:
        original_scores = -get_corpus_perplexity(original_texts, model, tokenizer, unit=unit, verbose=verbose)
        scores = scores - original_scores
        if 'cap' in comparison:
            scores = np.minimum(0, scores)
        elif 'abs' in comparison:
            scores = -np.abs(scores)
    return scores

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from ner_utils import get_ner_lists_smart_intersection
def get_intersections(slots_list, generated_seq, constraint_type):
        
    if slots_list is None or len(slots_list) == 0: 
        return [], 1
    
    if constraint_type == 'hard':
        hits = 0
        hits_list = []
        for slot in slots_list:
            slot_clean = slot.strip()
            if slot_clean.lower() in generated_seq.lower():
                hits += 1
                hits_list.append(slot_clean)
                
        return hits_list, hits / len(slots_list)
    
    elif constraint_type =='soft':
        interect_report, interect_score = get_ner_lists_smart_intersection(generated_seq, slots_list)
        return interect_report, interect_score
        
# previous name to look for in the gitlab repo is evaluate_formality_transfer
def evaluate_style_transfer(
    original_texts,
    rewritten_texts,
    slots_from_original = None,
    slots_constraint_type = 'soft', 
    refs=None,
    style_model_name='cointegrated/roberta-base-formality',
    style_model_tokenizer = None,
    style_target_label=1,
    meaning_model_name='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
    meaning_model_tokenizer = None,
    meaning_model_mounted = None,
    meaning_target_label='entailment',
    meaning_aggregation='prod',
    sts_model=None,
    cola_model_name='cointegrated/roberta-large-cola-krishna2020',
    cola_target_label=0,
    gpt_model_name='gpt2-medium',
    cola_weight=1,
    batch_size=32,
    verbose=True,
    aggregate=True,
    use_auth_token=None,
    sim_with_orig = True,
    classic_metric = False
):
    if verbose: print('Style evaluation')
    accuracy = evaluate_formality(
        rewritten_texts, 
        model=style_model_name, tokenizer = style_model_tokenizer, target_label=style_target_label, batch_size=batch_size, verbose=verbose, use_auth_token=use_auth_token
    )
    
    if sim_with_orig == True:
        if verbose: print('Meaning evaluation (with original)')  
        similarity = evaluate_meaning(
            original_texts, 
            rewritten_texts, 
            model_name=meaning_model_name, target_label=meaning_target_label, aggregation=meaning_aggregation, 
            batch_size=batch_size, verbose=verbose, use_auth_token=use_auth_token,
            model = meaning_model_mounted,
        )
    else:
        similarity = 0
        
    if sts_model:
        similarity2 = evaluate_meaning(
            original_texts, 
            rewritten_texts, 
            model_name=sts_model, bidirectional=False, target_label=None, aggregation=None,
            batch_size=batch_size, verbose=verbose, use_auth_token=use_auth_token,
        )
    else:
        similarity2 = 0

    cola, perplexity, fluency = None, None, None
    if cola_weight is not None:
        if verbose: print('Fluency evaluation')
        if cola_weight > 0:
            cola = evaluate_cola(
                rewritten_texts, 
                model_name=cola_model_name, batch_size=batch_size, verbose=verbose, use_auth_token=use_auth_token, target_label=cola_target_label,
            )
            fluency = cola

        if cola_weight < 1:
            perplexity = evaluate_perplexity(
                rewritten_texts,
                original_texts=original_texts,
                model_name=gpt_model_name, verbose=verbose, comparison='relative', use_auth_token=use_auth_token
            )
            fluency = perplexity
        if 0 < cola_weight < 1:
            fluency = cola ** cola_weight * sigmoid(perplexity) ** (1-cola_weight)
       
    if classic_metric == True:
        if verbose: print('BLEU evaluation')

        chrf_clc = CHRF()
        if aggregate:
            bleu_clc = BLEU()
            self_bleu = bleu_clc.corpus_score(hypotheses=rewritten_texts, references=[original_texts]).score
        else:
            bleu_clc = BLEU(effective_order = True)
            self_bleu = [bleu_clc.sentence_score(hyp, [ref]).score for hyp, ref in zip(rewritten_texts,original_texts)]
    else:
        self_bleu = None
        
    
    
    if refs is not None:
        if verbose: print('Meaning evaluation wrt reference')
        ref_meaning = evaluate_meaning_ref(
                refs, 
                rewritten_texts, 
                model_name=meaning_model_name, tokenizer = meaning_model_tokenizer, target_label=meaning_target_label, aggregation=meaning_aggregation, 
                batch_size=batch_size, verbose=verbose, use_auth_token=use_auth_token,
                model = meaning_model_mounted,
            )
        if classic_metric == True:

            if aggregate:
                ref_bleu = bleu_clc.corpus_score(hypotheses=rewritten_texts, references=refs).score
                ref_chrf = chrf_clc.corpus_score(hypotheses=rewritten_texts, references=refs).score
            else:
                ref_bleu = [bleu_clc.sentence_score(hyp, [ref]).score for hyp, ref in zip(rewritten_texts,refs)]
                ref_chrf = [chrf_clc.sentence_score(hyp, [ref]).score for hyp, ref in zip(rewritten_texts,refs)]
                
        else:
            ref_bleu = None
            ref_chrf = None
            
    else:
        ref_meaning = None
    
    slots_preservation = None
    if slots_from_original is not None:
        if verbose: print(f'Slots evaluation in <{slots_constraint_type}> mode')
        
        report_and_score_list = [get_intersections(slot_list, gener_txt, slots_constraint_type) for slot_list, gener_txt in tqdm(zip(slots_from_original, rewritten_texts), total = len(slots_from_original))]
        
        slots_preservation = np.array([el[1] for el in report_and_score_list])
        slots_preservation_report = [el[0] for el in report_and_score_list]
      
#     print(slots_preservation)
#     for i in range(len(rewritten_texts)):
#         if accuracy[i] is None:
#             print("acc")
#         if ref_meaning[i] is None:
#             print("ref_meaning")
#         if slots_preservation[i] is None:
#             print("slots_preservation", i)
    
    joint = accuracy * ref_meaning * slots_preservation
    
#     if verbose:
#         print(f'Style accuracy:       {np.round(np.mean(accuracy), 4)}')
#         print(f'Meaning preservation: {np.round(np.mean(similarity), 4)}')
#         print(f'Meaning, alternative: {np.round(np.mean(similarity2), 4)}')
#         print(f'CoLA fluency:         {np.round(np.mean(cola), 4) if cola is not None else "-"}')
#         print(f'GPT fluency:          {np.round(np.mean(perplexity), 4) if perplexity is not None else "-"}')
#         print(f'Joint fluency:        {np.round(np.mean(fluency), 4) if fluency is not None else "-"}')
#         print(f'Joint score:          {np.round(np.mean(joint), 4)}')
#         print(f'Self-BLEU:            {np.round(np.mean(self_bleu), 4)}')
#         print(f'Ref-BLEU:             {np.round(np.mean(ref_bleu), 4) if ref_bleu is not None else "-"}')
#         print(f'Ref-ChrF:             {np.round(np.mean(ref_chrf), 4) if ref_chrf is not None else "-"}')
#         print(f'Ref-Meaning:          {np.round(np.mean(ref_meaning), 4) if ref_meaning is not None else "-"}')
#         print(f'Slots-Preservation:          {np.round(np.mean(slots_preservation), 4) if slots_preservation is not None else "-"}')
    result = dict(
        accuracy=accuracy,
#         similarity=similarity,
#         similarity2=similarity2,
#         cola=cola,
#         perplexity=perplexity,
#         fluency=fluency,
        joint=joint,
#         self_bleu=self_bleu,
#         ref_bleu=ref_bleu,
#         ref_chrf=ref_chrf,
        ref_meaning=ref_meaning,
        slots_preservation=slots_preservation,
        slots_preservation_report = slots_preservation_report,
    )
    if aggregate:
        del result['slots_preservation_report']
        return {k: np.mean(v) if np.ndim(v) == 1 else v for k, v in result.items()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)
    parser.add_argument('-r', "--refs", help="path to reference sentences", default=None)
    parser.add_argument('-t', "--token", help="huggingface_token", default=None)
    parser.add_argument('-c', "--target_class", help="target class of the prediction", default=1)
    parser.add_argument('-s', "--style_model", help="the model for style prediction", default="SkolkovoInstitute/roberta-base-formality-ranker-v1")
    args = parser.parse_args()


    with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
        inputs = [line.strip() for line in input_file.readlines()]
        preds = [line.strip() for line in preds_file.readlines()]
    if len(inputs) != len(preds):
        raise ValueError(f'Inputs and preds have different lenghts ({len(inputs)} vs {len(preds)})')
    refs = None
    if args.refs:
        with open(args.refs, 'r') as refs_file:
            refs = [
                [x.strip() for x in line.split("[REFSEP]")]
                for line in refs_file.readlines()
            ]
        if len(inputs) != len(refs):
            raise ValueError(f'Inputs and refs have different lenghts ({len(inputs)} vs {len(refs)})')
        refs = transpose_refs(refs)

    result = evaluate_style_transfer(
        inputs,
        preds,
        refs=refs,
        use_auth_token=args.token,
        style_target_label=args.target_class,
        style_model_name=args.style_model,
        sts_model='cross-encoder/stsb-roberta-base',
        verbose=True,
    )
    print(result)
