import json
import os
import pickle
from PIL import Image

from collections import Counter
from torch.utils.data import Dataset

import utils
import torch

import time

class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, train, num_samples, dictionaries=None, transform=None, use_qa=False, **kwargs):
        """
        Args:
        clevr_dir (string): Root directory of CLEVR dataset
        train (bool): Tells if we are loading the train or the validation datasets
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.num_samples = num_samples
        self.use_qa = use_qa
        self.split = "train" if train else "val"
        self.dirname = os.path.dirname(__file__)
        
        if train:
            quest_json_filename = os.path.join(self.dirname, clevr_dir, 'questions'
                                               , 'CLEVR_train_questions.json') if self.use_qa else None
            scene_json_filename = os.path.join(self.dirname, clevr_dir, 'scenes', 'CLEVR_train_scenes.json')
            self.img_dir = os.path.join(self.dirname, clevr_dir, 'images', 'train')
        else:
            quest_json_filename = os.path.join(self.dirname, clevr_dir, 'questions'
                                               , 'CLEVR_val_questions.json') if self.use_qa else None
            scene_json_filename = os.path.join(self.dirname, clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
            self.img_dir = os.path.join(self.dirname, clevr_dir, 'images', 'val')
        

        if self.use_qa:
            cached_questions = quest_json_filename.replace('.json', '.pkl')
            if os.path.exists(cached_questions):
                print('==> using cached questions: {}'.format(cached_questions))
                with open(cached_questions, 'rb') as f:
                    self.questions = pickle.load(f)
            else:
                with open(quest_json_filename, 'r') as json_file:
                    self.questions = json.load(json_file)['questions']
            with open(cached_questions, 'wb') as f:
                pickle.dump(self.questions, f)
                
            self.dictionaries = dictionaries

        cached_scenes = scene_json_filename.replace('.json', '.pkl')    
        if os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                self.objects = pickle.load(f)
        else:
            all_scene_objs = []
            with open(scene_json_filename, 'r') as json_file:
                scenes = json.load(json_file)['scenes']
                print('caching all objects in all scenes...')
                for s in scenes:
                    objects = s['objects']
                    objects_attr = []
                    for obj in objects:
                        attr_values = []
                        for attr in sorted(obj):
                        # convert object attributes in indexes
                            if attr in utils.classes:
                                attr_values.append(utils.classes[attr].index(obj[attr])+1) #zero is reserved for padding
                            else:
                                '''if attr=='rotation':
                                attr_values.append(float(obj[attr]) / 360)'''
                                if attr=='3d_coords':
                                    attr_values.extend(obj[attr])
                                    objects_attr.append(attr_values)
                    all_scene_objs.append(torch.FloatTensor(objects_attr))
            self.objects = all_scene_objs
            with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)

        self.objects = self.objects[:self.num_samples]
        self.clevr_dir = clevr_dir
        self.transform = transform
        

#     def answer_weights(self):
#         n = float(len(self.questions))
#         answer_count = Counter(q['answer'].lower() for q in self.questions)
#         weights = [n/answer_count[q['answer'].lower()] for q in self.questions]
#         return weights

    def __len__(self):
        return self.num_samples
#         return len(self.questions)

    
    def __getitem__(self, idx):


        if self.use_qa:
            current_question = self.questions[idx]
            scene_idx = current_question['image_index']
            
            obj = self.objects[scene_idx]
            img_filename = os.path.join(self.img_dir, current_question['image_filename'])
            image = Image.open(img_filename).convert('RGB')

            question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
            answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
            
            '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''
            
            sample = {'images': image, 'question': question, 'answer': answer
#                       , 'obj':obj # cannot be batched
                     }

        else:
            padded_index = str(idx).rjust(6, '0')
            img_filename = os.path.join(self.dirname, self.img_dir, 'CLEVR_{}_{}.png'.format(self.split,padded_index))
            scene_idx = idx
            
            obj = self.objects[scene_idx]
            image = Image.open(img_filename).convert('RGB')
            
            sample = {'images': image
#                       , 'obj':obj # cannot be batched
                     }
            

        if self.transform:
            sample['images'] = self.transform(sample['images'])

        return sample