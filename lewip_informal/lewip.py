from transformers import (AutoModelForSeq2SeqLM, 
                          AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          pipeline)
import torch

from .utils import preprocess_text, convert_template_to_t5

MASK_TEMPLATE = " <extra_id_{}> "
import re

class LEWIP:
    
    def __init__(self, predefined_entities = True, use_cuda = False, 
                 model_name = "SkolkovoInstitute/LEWIP-informal",
                tagger_model_name='SkolkovoInstitute/LEWIP-informal-tagger'):
        
        if use_cuda == True:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.tagger_model_name = tagger_model_name
        if predefined_entities == False:
            self.initialize_tagger()
        else:
            self.tagger_pipe = None

    def initialize_tagger(self, tagger_model_name=None):
        
        if tagger_model_name is None: tagger_model_name = self.tagger_model_name
            
        tagger_model = AutoModelForTokenClassification.from_pretrained(tagger_model_name)
        self.tagger_tokenizer = AutoTokenizer.from_pretrained(tagger_model_name)  
        self.tagger_pipe = pipeline(
                                "token-classification",
                                model=tagger_model,
                                tokenizer=self.tagger_tokenizer,
                                framework="pt",
                                device=0,
                                aggregation_strategy="max"
                            )
        
    def generate(self,text, important_entities = None, show_template = False):
        
        if important_entities is None and self.tagger_pipe is None:
            print("""Looks like you forgot to set 'predefined_entities' variable to False when initializing the class.
                  The tagger is being initialized now...""")
            self.initialize_tagger()
        
        text = preprocess_text(text)
        template = self.get_template(text, important_entities)
        if show_template == True:
            print(template)
        
        input_ids = self.tokenizer(
            text,
            text_pair=template,
            add_special_tokens=True,
            max_length=200,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            ).input_ids.to(self.model.device)
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=5,
            num_return_sequences=1,
            max_length=300,
            repetition_penalty=2.5
        )
        
        for sample_output_ids in output_ids:
            fillers = self.tokenizer.decode(sample_output_ids, skip_special_tokens=False)
            mask_count = template.count("extra_id")
            target = template
            for mask_num in range(mask_count):
                current_mask = MASK_TEMPLATE.format(mask_num).strip()
                next_mask = MASK_TEMPLATE.format(mask_num + 1).strip()
                start_index = fillers.find(current_mask) + len(current_mask)
                end_index = fillers.find(next_mask)
                filler = fillers[start_index:end_index]
                target = target.replace(current_mask, filler)
            target = " ".join(target.split())
            target = target.replace(" ,", ",")
            
        return target
    
    def get_template(self, text, important_entities_list = None):
        
        if important_entities_list is None:
            template = self.get_template_from_tagger(text)
        else:
            template = self.get_template_from_predefined(text, important_entities_list)
            
        return template

    def get_template_from_tagger(self, text):
        
        tagged_sequence = self.tagger_pipe(text, batch_size=1)
        template = []
        
        for group in tagged_sequence:
            tag = group["entity_group"]
            phrase = group["word"]
            pad_index = phrase.find(self.tagger_tokenizer.pad_token)
            if pad_index != -1:
                phrase = phrase[:pad_index]
            if tag == "delete":
                continue
            if tag == "replace":
                phrase = self.tagger_tokenizer.mask_token
            template.append(phrase.strip())
        template = " ".join(template)

        template = convert_template_to_t5(template, self.tagger_tokenizer.mask_token)
        
        return template
        
    def get_template_from_predefined(self, text, important_entities_list):
    
        s_index = 0
        slot_dict = {}
        for slot in important_entities_list:
            backslah_slot = re.sub("\+",'\+',slot)
            text = re.sub(backslah_slot, f' M{s_index} ', text)
            slot_dict[f'M{s_index}'] = slot
            s_index += 1

        text_list = text.split()
        template = ''
        prev_is_slot = False

        for t in text_list:
            if prev_is_slot == True:
                template += "_"
    #             extra_id_idx += 1

            if re.match('M\d',t):
                template += slot_dict[t] 
                prev_is_slot = True
            else:
                template += '_'
                prev_is_slot = False

        template = re.sub('_+',' <extra_id_> ', template)

        template_fin = []
        extra_id_idx = 0
        for t in template.split():
            if t == '<extra_id_>':
                template_fin.append(f'<extra_id_{extra_id_idx}>')
                extra_id_idx += 1
            else:
                template_fin.append(t)

        template_fin = ' '.join(template_fin)

        return template_fin
        