import os
import pandas as pd

from sklearn.metrics import accuracy_score

def zero_rule_algorithm_classification(train, test):
	output_values = [row for row in train['Source CM']]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

train = pd.read_csv('../data/metaphor-scm-train-part1.csv')
test = pd.read_csv('../data/metaphor-scm-test-part1.csv')

y_pred = test['Source CM'].tolist()
y_test = zero_rule_algorithm_classification(train,test)

acc = accuracy_score(y_test,y_pred)


