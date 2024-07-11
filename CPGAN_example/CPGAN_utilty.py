from functools import reduce

def compute_one_zero_acc(labels,test_label):
    correct=0
    for i in range(len(labels)):
        if test_label[i]==labels[i]:
            correct=correct+1
    acc=correct/len(labels)
    return acc

def training_curry(*fns):
    def inner(x):
        return reduce(lambda v, f: f(v), fns, x)
    return inner