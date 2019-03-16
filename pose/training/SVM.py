import argparse
import os
import pickle
import shutil

from sklearn import svm
from sklearn.metrics import accuracy_score

from ReadData import read_data_files, split_data

def parse_args():
	parser = argparse.ArgumentParser(description = 'Train SVM')
	# You have to give individual csv files based on the order in which position id of poses are made.
	parser.add_argument('csv_files', help = 'Comma separated list of paths of training files', type = str)
	parser.add_argument('--output-path', dest = 'output_path', type = str, default = None,
						help = 'Path of folder where to store the trained model')
	parser.add_argument('--train-test-split', dest = 'train_test_split', default = 0.85, type = float,
					    help = 'Ratio of train to test dataset(0-1)')
	parser.add_argument('--max-samples', dest = 'max_samples', type = int, default = 800,
						help = 'Maximum number of samples per class allowed')
	args = parser.parse_args()
	return args

def apply_svc(train_data, test_data, output_at):
	X_train, y_train = train_data
	X_test, y_test = test_data

	clf = svm.SVC()
	clf.fit(X_train, y_train)  

	y_test_pred = clf.predict(X_test)
	test_acc = accuracy_score(y_test, y_test_pred)
	print('SVC Test accuracy: {:.5f}'.format(test_acc))

	shutil.rmtree(output_at, ignore_errors = True)
	os.mkdir(output_at)
	with open('{}/svc.pickle'.format(output_at), 'wb') as handle:
		pickle.dump(clf, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	args = parse_args()

	data_files = [os.path.abspath(p.strip()) for p in args.csv_files.split(',')]
	X_data, y_data = read_data_files(data_files, args.max_samples)
	train_data, test_data = split_data(X_data, y_data, args.train_test_split)

	# If output folder is None, will create an output folder
	# Also, the output folder will be deleted and recreated again.
	output_at = 'Checkpoint' if args.output_path is None else os.path.abspath(args.output_path)

	apply_svc(train_data, test_data, output_at)