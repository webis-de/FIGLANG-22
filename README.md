# FIGLANG-22

This repository contains the code for our paper 'Back to the Roots:Predicting the Source Domain of Metaphors using Contrastive Learning' accepted at the Figurative Language workshop held in conjunction with EMNLP '22.  

ABSTRACT: Metaphors frame a given target domain using concepts from another, usually more concrete, source domain. Previous research in NLP has focused on the identification of metaphors and the interpretation of their meaning. In contrast, this paper studies to what extent the source do-main can be predicted computationally from a metaphorical text. Given a dataset with metaphorical texts from a finite set of source domains, we propose a contrastive learning approach that ranks source domains by their likelihood of being referred to in a metaphorical text. In experiments, it achieves reasonable performance even for rare source domains, clearly outperforming a classification baseline.

Instructions to run:

```
pip -r requirements.txt
```

To run the contrastive learning approach:

```
cd Contrastive Learning/code/
python3 run.py 
```

and

```
python3 run_splits.py (to run the data splits) 
```

To run classification approach: 

```
cd Classification/code/
python3 run_bert.py (to run with bert as the encoder)
python3 run_distilbert.py (to run with distilbert as the encoder)
```

