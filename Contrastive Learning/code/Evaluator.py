from base64 import encode
from posixpath import split
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from aspects import source_cm
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO)

class RankingEvaluator(SentenceEvaluator):
    def __init__(
        self,  
        anchors_with_ground_truth_candidates,
        task='scm',
        main_distance_function: SimilarityFunction = None, 
        name: str = '', 
        batch_size: int = 32, 
        show_progress_bar: bool = False, 
        write_csv: bool = True,
        loss: str = 'MultipleNegativesRankingLoss',
        split: int = 0,
        iterations: int=0
        ): 

        self.anchors_with_ground_truth_candidates = anchors_with_ground_truth_candidates
        self.source_cm = source_cm
        self.batch_size = batch_size
        self.show_progress_bar=show_progress_bar
        self.task = task
        self.loss = loss
        self.split = split
        self.iterations = iterations

    def encode(self,model,anchors,candidates):
        self.model = model
        embeddings_anchors = self.model.encode(
            anchors, batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_candidates = self.model.encode(
            candidates,  
            convert_to_numpy=True
        )

        
        embeddings_scm = self.model.encode(
            self.source_cm,  
            convert_to_numpy=True
        )

        assert len(embeddings_anchors) == len(embeddings_candidates)

        assert len(anchors) == len(embeddings_anchors)
        self.ANCHORS_WITH_EMBEDDINGS = dict(zip(anchors,embeddings_anchors))

        assert len(candidates) == len(embeddings_candidates)
        self.CANDIDATES_WITH_EMBEDDINGS = dict(zip(candidates,embeddings_candidates))
        
        self.SCM_WITH_EMBEDDINGS = dict(zip(self.source_cm,embeddings_scm))

        print('\nLength of list of generated embeddings is equal to the length of the list of items\n')

        return (self.ANCHORS_WITH_EMBEDDINGS,self.CANDIDATES_WITH_EMBEDDINGS, self.SCM_WITH_EMBEDDINGS)

    def compare(self,anchor, anchor_embedding,ground_truth_embeddings):
        result = dict()
        for candidate,candidate_embedding in ground_truth_embeddings.items():
            assert len(anchor_embedding) == len(candidate_embedding)
            distance = paired_cosine_distances(anchor_embedding.reshape(1,-1), candidate_embedding.reshape(1, -1))
            if anchor not in result.keys():
                result[anchor] = [(candidate,distance[0])]
            else:
                result[anchor].append((candidate,distance[0]))
            
        return result

    def get_most_similar_candidate(self,anchor_distance_scores):
        self.anchor_distance_scores = anchor_distance_scores
        candidate_scores = next(iter(self.anchor_distance_scores.values()))
        '''
        print('\n')
        print(anchor)
        print('\n')
        print(candidate_scores)
        print('\n')
        '''
        most_similar_candidate = min(candidate_scores, key = lambda t: t[1])
        
        return most_similar_candidate[0]

    def get_all_anchors_most_similar_candidate(self,anchor_with_embeddings,SCM_WITH_EMBEDDINGS):
        result = dict()
        for anchor,embedding in anchor_with_embeddings.items():
            anchor_distance_scores = self.compare(anchor, embedding,SCM_WITH_EMBEDDINGS)
            most_similar_candidate = self.get_most_similar_candidate(anchor_distance_scores)
            result[anchor] = most_similar_candidate

        return result

    def __call__(self,model,output_path: str = None, epoch: int = -1, steps: int = -1):

        self.anchors = list(); self.candidates = list(); self.epoch=epoch; self.steps=steps

        if self.epoch != -1:
            if self.steps == -1:
                out_txt = " after epoch {}:\n".format(self.epoch)
            else:
                out_txt = " in epoch {} after {} steps:\n".format(self.epoch, self.steps)
        else:
            out_txt = ":\n"

        logging.info("\nEvaluating the model"+out_txt)
        print(len(self.anchors_with_ground_truth_candidates))
        for anchor, candidate in self.anchors_with_ground_truth_candidates.items():
            self.anchors.append(anchor)
            self.candidates.append(candidate)
        assert len(self.anchors) == len(self.candidates)
        anchor_with_embeddings, candidate_with_embeddings, SCM_WITH_EMBEDDINGS = self.encode(model, self.anchors,self.candidates)

        '''
        print(next(iter(anchor_with_embeddings.items())))
        print('\n')

        print(next(iter(candidate_with_embeddings.items())))
        print('\n')
        
        print(next(iter(SCM_WITH_EMBEDDINGS.items())))
        print('\n')
        '''

        print('\nDictionary with anchors,candidates and their respective embeddings have been created\n')

        final = dict()
        results = dict()
        most_similar_candidates = self.get_all_anchors_most_similar_candidate(anchor_with_embeddings,SCM_WITH_EMBEDDINGS)
        
        for anchor, true in self.anchors_with_ground_truth_candidates.items():
            predicted = most_similar_candidates[anchor]
            if true not in final.keys():
                if predicted == true:
                    final[true] = {'correct':1,'total':1}
                else:
                    final[true] = {'correct':0,'total':1}
            else:
                if predicted == true:
                    final[true]['correct']+=1
                    final[true]['total']+=1
                else:
                    final[true]['total']+=1

        for label, content in final.items():
            correct = content['correct']
            total = content['total']
            acc = (correct/total)*100
            results[label]=(correct,total,acc)

        sum_correct=0; sum_total=0
        for label, content in final.items():
            correct = content['correct']
            total = content['total']
            sum_correct+=correct
            sum_total+=total

        macro_average = sum_correct/sum_total

        logging.info("Epoch:   \t{:.2f}".format(self.epoch))
        logging.info("Steps:   \t{:.2f}".format(self.steps))
        logging.info("Macro Accuracy:   \t{:.2f}".format(macro_average))

        eval_metrics = {"accuracy": macro_average, "split": self.split}
        wandb.log(eval_metrics)

        return macro_average
