from __future__ import division
import nltk
import os
import numpy
import random
from nltk.corpus import brown
from nltk.corpus import indian
from pickle import dump
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from scipy import interp
from itertools import cycle
from sklearn.linear_model import LogisticRegression
import string
import math
#tokenize = lambda doc: doc.split(" ")

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
  #  tokenized_documents = [d for (d,c) in documents]
    idf = inverse_document_frequencies(documents)
    tfidf_documents = []
    
    #labels = [c for (d,c) in documents]
    for document in documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            tid=tf*idf[term]
            doc_tfidf.append(tid)
            #print("term=",term,"tfidf=",tid)
        tfidf_documents.append(doc_tfidf)
        
    return tfidf_documents


def transformDataset(sentences):
    wordFeatures = []
    wordLabels = []
    for sent in sentences:
       wordFeatures.append(document_features(sent))
       wordLabels.append(sent[1])
    return wordFeatures, wordLabels



def transformDataset1(sentences):
    wordFeatures = []
    wordLabels = []
    for sent in sentences:
       wordFeatures.append(document_features(sent))
       wordLabels.append(sent[1])
    return wordFeatures, wordLabels
def document_features(document): 
    document_words = [w for w in document]
    #print(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
#print(document_features(nltk.corpus.indian.words("/prem/1.txt")))

def trainDecisionTree(trainFeatures, trainLabels):
    clf = make_pipeline( OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy')))
    
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean(),scores
def trainkNeighbour(trainFeatures, trainLabels):
    print("k clss")
    clf = make_pipeline( KNeighborsClassifier(n_neighbors=10))
    #clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean(),scores
def trainNaiveBayes(trainFeatures, trainLabels):
    clf = make_pipeline(MultinomialNB())
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean(),scores
def trainNN(trainFeatures, trainLabels):
    clf = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1))
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean(),scores
def trainSvc(trainFeatures, trainLabels):
    print("Support Vector Machine")
    clf = make_pipeline(SVC())
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    print("class svc comp")
    return clf, scores.mean(),scores
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def createList(foldername, fulldir = True, suffix=".jpg"):
    file_list_tmp = os.listdir(foldername)
    #print len(file_list_tmp)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list

file=createList("marathi_poems/prem",suffix=".txt")

file1=createList("marathi_poems/vir",suffix=".txt")

file2=createList("marathi_poems/joy",suffix=".txt")

file3=createList("marathi_poems/fear",suffix=".txt")


file4=createList("marathi_poems/sadness",suffix=".txt")
file5=createList("marathi_poems/wonder",suffix=".txt")
file6=createList("marathi_poems/anger",suffix=".txt")


file7=createList("marathi_poems/depression",suffix=".txt")
file8=createList("marathi_poems/peace",suffix=".txt")

stop=open("marathi_stopwords.txt","r",encoding='utf-8')
words=stop.read()
word=nltk.word_tokenize(words)
print(word)
documents=[]
all_words=[]
for fname in file:
    a=list(nltk.corpus.indian.words(fname))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"prem")])
    all_words.extend(b)

for fname1 in file1:
    a=list(nltk.corpus.indian.words(fname1))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"vir")])
    all_words.extend(b)
for fname2 in file2:
    a=list(nltk.corpus.indian.words(fname2))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"joy")])
    all_words.extend(b)
for fname3 in file3:
    a=list(nltk.corpus.indian.words(fname3))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"fear")])
    all_words.extend(b)


for fname4 in file4:
    a=list(nltk.corpus.indian.words(fname4))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"sadness")])
    all_words.extend(b)   
for fname5 in file5:
    a=list(nltk.corpus.indian.words(fname5))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"wonder")])
    all_words.extend(b)
for fname6 in file6:
    a=list(nltk.corpus.indian.words(fname6))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"anger")])
    all_words.extend(b)


for fname7 in file7:
    a=list(nltk.corpus.indian.words(fname7))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"depression")])
    all_words.extend(b)   
for fname8 in file8:
    a=list(nltk.corpus.indian.words(fname8))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"peace")])
    all_words.extend(b)
#print("total words")
#print(all_words)


#all_words_new = nltk.FreqDist(all_words)
#print(all_words_new)
#print(len(all_words_new))
#word_features = list(all_words_new)[:1000]

#print("indian hfhsdjfjfjd")
random.shuffle(documents)
size = int(len(documents) * 0.7)
tags = [tag for (document, tag) in documents]
train_sents = documents[:size]
#print(len(train_sents))
test_sents = documents[size:]
#trainFeatures, trainLabels = transformDataset(train_sents)
#testFeatures, testLabels = transformDataset(test_sents)
corpus=[d for (d,c) in documents]
labels=[c for (d,c) in documents]
features=tfidf(corpus)

#print(features[1])

#features,labels=transformDataset(documents)
#vec = DictVectorizer()
#features_new=vec.fit_transform(features).toarray()
#print(features_new.shape)

print(len(features))
print(len(labels))
clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1).fit(features, labels)
clf = clf.fit(features, labels)
model = SelectFromModel(clf, prefit=True)
fe = model.transform(features)
#print(fit.scores_)
print(fe.shape)

# summarize selected features
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(fe,labels, test_size=0.33, random_state=42)


print("length of testLabels=",len(testLabels))
#for l in testLabels:
#    print("label=",l)
#print("features=",trainFeatures[1],"label=",trainLabels[1])
#featuresets = [(document_features(d), c) for (d,c) in documents]

var = 1
while var == 1:
    print("******************MENU********************")
    print("case 1:Naive Bayes classifier")
    print("case 2: Decision tree classifier")
    print("case 3: Neural network")
    print("case 4: K nearest neighbour")
    print("case 5: Support Vector Machine")
    print("case 6: exit")   
    print("enter your choice")
    ch=input()
    if ch == "6":
        
        var = 2
        continue
    elif ch == "1":
        print("naive bayes")
        tree_model, tree_model_cv_score,scores = trainNaiveBayes(trainFeatures, trainLabels)
        
        
    elif ch =="2":
        tree_model, tree_model_cv_score,scores = trainDecisionTree(trainFeatures, trainLabels)
    elif ch == "3":
        tree_model, tree_model_cv_score,scores = trainNN(trainFeatures, trainLabels)
    elif ch == "4":
        tree_model, tree_model_cv_score,scores = trainkNeighbour(trainFeatures, trainLabels)
    else:
          tree_model, tree_model_cv_score,scores = trainSvc(trainFeatures, trainLabels)
    Max = 0
    for i in range(1,6):
        print("accuracy in fold ",i, " =",scores[i-1])
        
        
    print("accuracy on train data  = ")
    print(tree_model_cv_score)
    print("accuracy on test data  = ")
    print(tree_model.score(testFeatures, testLabels))
    y_pred = tree_model.fit(trainFeatures, trainLabels).predict(testFeatures)
    print("y_predicted=",y_pred)    
    print("y_pred unique= ",np.unique(y_pred))
    print("y_test unque=",np.unique(testLabels))

#scatter plot
#plt.figure()
#plt.scatter(testLabels, y_pred)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
#end


    cnf_matrix = confusion_matrix(testLabels, y_pred)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(tags), normalize=True,title='Normalized confusion matrix')


    random_state = np.random.RandomState(0)
    classes=['anger','depression', 'fear', 'joy', 'peace', 'prem', 'sadness', 'vir', 'wonder']
    te_lab = label_binarize(testLabels,classes )
    print("label",te_lab[0],"textLabel",testLabels[i])
    exit(0)
    y_p=label_binarize(y_pred,classes )

#print("len of te_lab",len(te_lab))
#print("len of y_pr",len(y_p))

    precision = dict()
    recall = dict()
    ther = dict()
    average_precision = dict()


    print(classification_report(te_lab, y_p, target_names=classes))
           

    for i in range(0,len(classes)):
        precision[i], recall[i], ther[i] = precision_recall_curve(te_lab[:, i],
                                                        y_p[:, i])
        average_precision[i] = average_precision_score(te_lab[:, i], y_p[:, i])
#for i in range(0,len(classes)):
#    print("Precision of",classes[i],"=",precision[i])
#    print("Recall of",classes[i],"=",recall[i])
#    print("Threshold of",classes[i],"=",ther[i])
    
# A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(te_lab.ravel(),y_p.ravel())
    average_precision["micro"] = average_precision_score(te_lab, y_p,average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))

    plt.show()

    if(ch=="1"):
        plt.figure() 
        param_range = np.arange(1, 21, 2)
        print(tree_model.get_params().keys())
        train_scores, validation_scores = validation_curve(tree_model, features, labels,param_name='multinomialnb__alpha',param_range=param_range)
        plt.title("Validation Curve with Naive Bayes")
        plt.xlabel("alpha")
        plt.ylabel("Score")
        plt.plot(param_range, validation_scores.mean(axis=1), label='cross-validation')
        plt.plot(param_range, train_scores.mean(axis=1), label='training')
        plt.legend(loc='best')
        plt.show()
    elif (ch=="2"):
        print(tree_model.get_params().keys())
        plt.figure() 
        param_range = np.arange(1, 20, 2)
        #print(tree_model.get_params().keys())
        train_scores, validation_scores = validation_curve(tree_model, features, labels,param_name='onevsrestclassifier__estimator__min_impurity_split',param_range=param_range)
        plt.title("Validation Curve with Decision tree")
        plt.xlabel("min impurity split")
        plt.ylabel("Score")
        plt.plot(param_range, validation_scores.mean(axis=1), label='cross-validation')
        plt.plot(param_range, train_scores.mean(axis=1), label='training')
        plt.legend(loc='best')
        plt.show()
    elif(ch=="3"):
        plt.figure() 
        param_range = np.arange(1, 101, 2)
        print(tree_model.get_params().keys())
        train_scores, validation_scores = validation_curve(tree_model, features, labels,param_name='mlpclassifier__hidden_layer_sizes',param_range=param_range)
        plt.title("Validation Curve with Neural Network")
        plt.xlabel("hidden layer sizes")
        plt.ylabel("Score")
        plt.plot(param_range, validation_scores.mean(axis=1), label='cross-validation')
        plt.plot(param_range='best')
        plt.show()
    elif(ch=="4"):
        plt.figure() 
        param_range = np.arange(1, 12,1)
        print(tree_model.get_params().keys())
        
        train_scores, validation_scores = validation_curve(tree_model, features,labels,param_name='kneighborsclassifier__n_neighbors',param_range=param_range)
        plt.title("Validation Curve with KNN")
        plt.xlabel("n_neighbours")
        plt.ylabel("Score")
        plt.plot(param_range, validation_scores.mean(axis=1), label='cross-validation')
        plt.plot(param_range, train_scores.mean(axis=1), label='training')
        plt.legend(loc='best')
        plt.show()
    else:
        plt.figure() 
        param_range = np.logspace(-6, -1, 5)
        print(tree_model.get_params().keys())
        train_scores, validation_scores = validation_curve(tree_model, features, labels,param_name='svc__gamma',param_range=param_range)
        plt.title("Validation Curve with SVM")
        plt.xlabel("gamma")
        plt.ylabel("Score")
        plt.plot(param_range, validation_scores.mean(axis=1), label='cross-validation')
        plt.plot(param_range, train_scores.mean(axis=1), label='training')
        plt.legend(loc='best')
        plt.show()
        
    plt.figure()
    train_sizes, train_scores, validation_scores = learning_curve(tree_model, features ,labels, train_sizes=np.logspace(-1, 0, 20))
    plt.xlabel('Trainging Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.plot(train_sizes, validation_scores.mean(axis=1), label='cross-validation')
    plt.plot(train_sizes, train_scores.mean(axis=1), label='training')
    plt.legend(loc='best')
    plt.show()    

    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range (0,len(classes)):
        fpr[i], tpr[i], _ = roc_curve(te_lab[:, i], y_p[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(te_lab.ravel(), y_p.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#Plot of a ROC curve for a specific class
    #for i in range (0,len(classes)):
    #    plt.figure()
    #    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #    plt.plot([0, 1], [0, 1], 'k--')
    #    plt.xlim([0.0, 1.0])
    #    plt.ylim([0.0, 1.05])
    #    plt.xlabel('False Positive Rate')
    #    plt.ylabel('True Positive Rate')
    #    tit='Receiver operating characteristic example ='+classes[i]
    #    plt.title(tit)
     #   plt.legend(loc="lower right")
     #   plt.show()

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink' , linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','yellow'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()








