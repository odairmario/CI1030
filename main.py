import argparse
import datetime
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, confusion_matrix, f1_score,
                             recall_score, roc_curve)
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

np.set_printoptions(threshold=sys.maxsize)


class Labels(object):

    ROTULE_SUSPECT = 1
    ROTULE_CORRECT = 0
    # features
    FEATURE_IntervaloIP = 1
    FEATURE_formatInvalido = 2
    FEATURE_uaInvalido = 3
    FEATURE_typeMethod = 4
    FEATURE_cgnat = 5
    FEATURE_ipsDiferentes = 6
    FEATURE_metaUser = 7

    # Features values
    features_values = {
        # feature format
        "invalid_format": 1,
        "valid_format": 0,
        # feature ua
        "valid_ua": 0,
        "invalid_ua": 1,
        # feature cgnat
        "valid_cgnat": 0,
        "invalid_cgnat": 1,
        # feature typeMethod
        "get_method": 0,
        "post_method": 1,
        "unkow_method": 2,
        # ipsDiferentes
        "valid_ip_diferent": 0,
        "invalid_ip_diferent": 1,
        "valid_metauser": 0,
        "invalid_metauser":1,
    }


class Data(object):

    """Class to collect data and select features"""

    def __init__(self,**config):
        """TODO: to be defined.

        :**config: TODO

        """
        self._raw_data_path = config.get("raw_data_path")
        self._attributes = config.get("attributes")

        self._raw_data = []
        self._data = []
    def load(self):
        """TODO: Docstring for load.
        :returns: TODO

        """
        with open(self._raw_data_path,"r") as _file:
            self._raw_data = json.load(_file)
    def __iter__(self):
        """TODO: Docstring for __iter__.
        :returns: TODO

        """

        return iter(self._data)

    def select_attributes(self, *attributes):
        """TODO: Docstring for select_attributes.

        :*attributes: TODO
        :returns: TODO

        """
        data = []

        attrit = attributes if attributes else self._attributes


        for _log in self._raw_data:
            data.append({k:_log[k] for k in attrit if k in _log} )

        self._data = data
    def print(self):
        """TODO: Docstring for print.
        :returns: TODO

        """
    def suffle(self):
        """TODO: Docstring for suffler.
        :returns: TODO

        """
        random.shuffle(self._data)
    def rotule(self):
        """TODO: Docstring for rotule.
        :returns: TODO

        """

        for data in self._data:

            if ((data["format"] is None ) or data.get("format","").find('*/*'))  or ( data["ua"] == None or("python" in data.get("ua",""))):
                data["label"] = Labels.ROTULE_SUSPECT
            else:
                data["label"] = Labels.ROTULE_CORRECT
    def split(self, train):
        """TODO: Docstring for split.

        :train: TODO
        :test: TODO
        :returns: TODO

        """

        return self._data[:train], self._data[train:]


class NormalizeData(object):

    """Docstring for NormalizeData. """


    def __init__(self,data,base_path):
        """TODO: to be defined.

        :data: TODO

        """
        self._base_path = base_path
        self._data = data
        self._remote_ip_is_mapped = False
        self._df = None
    def get_ua(self, ua):
        """TODO: Docstring for get_ua.

        :ua: TODO
        :returns: TODO

        """

        if ua is None or ua.find("python-requests"):
            return Labels.features_values.get("invalid_ua")

        return Labels.features_values.get("valid_ua")

    def get_intervalo_ip(self, remote_ip ):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """

        d = {"timer":[],"IP":[]}

        if self._remote_ip_is_mapped is False:

            for i,data in enumerate(self._data):

                timer = datetime.datetime.strptime(data["time"], "%Y-%m-%dT%H:%M:%S.%fZ")
                #timer = timer.strftime("%Y-%m-%dT%H:%M") # Por minuto
                timer = timer.strftime("%Y-%m-%dT%H") # Por hora

                d["timer"].append(timer)
                d["IP"].append(data["remote_ip"])
            self._df = pd.DataFrame(data=d)
            self._remote_ip_is_mapped = True
        df = self._df.loc[self._df["IP"] == remote_ip]
        series = df.groupby("timer").count()

        return int(series.mean()[0])

        #data = [ t["time"] for t in filter(lambda c: c["remote_ip"] == remote_ip,self._data) ]

    def get_format(self, _format):
        """TODO: Docstring for get_format.

        :_format: TODO
        :returns: TODO

        """

        if _format is None or _format.find("*/*"):
            return Labels.features_values.get("invalid_format")

        return Labels.features_values.get("valid_format")

    def get_type_method(self, typemethod):
        """TODO: Docstring for get_type_method.

        :typemethod: TODO
        :returns: TODO

        """

        if typemethod.find("GET"):
            return Labels.features_values.get("get_method")
        elif typemethod.find("POST"):
            return Labels.features_values.get("post_method")
        else:
            return Labels.features_values.get("unkow_method")

    def get_cgnat(self, remote_ip):
        """TODO: Docstring for get_cgnat.

        :remote_ip: TODO
        :returns: TODO

        """
        cgnat = "100.64"

        if remote_ip.find(cgnat):
            return Labels.features_values.get("valid_cgnat")

        return Labels.features_values.get("invalid_cgnat")
    def get_ip_diferentes(self, remote_ip,meta_ip):
        """TODO: Docstring for get_ip_diferentes.

        :remote_ip: TODO
        :meta_ip: TODO
        :returns: TODO

        """

        if remote_ip == meta_ip:
            return Labels.features_values.get("valid_ip_diferent")

        return Labels.features_values.get("invalid_ip_diferent")

    def get_meta_user(self, metauser):
        """TODO: Docstring for get_meta_user.

        :user: TODO
        :returns: TODO

        """

        if metauser is None:
            return Labels.features_values.get("invalid_metauser")

        return Labels.features_values.get("valid_metauser")
    def normalize(self):
        """TODO: Docstring for normalize.
        :returns: TODO

        """
        normalized_data = []

        for data in self._data:
            normalized_data.append({
                Labels.FEATURE_uaInvalido: self.get_ua(data.get("ua")),
                Labels.FEATURE_IntervaloIP: int(self.get_intervalo_ip(data.get("remote_ip"))),
                Labels.FEATURE_formatInvalido: self.get_format(data.get("format","")),
                Labels.FEATURE_typeMethod: self.get_type_method(data.get("typeMethod","")),
                Labels.FEATURE_cgnat: self.get_cgnat(data.get("remote_ip")),
                Labels.FEATURE_ipsDiferentes: self.get_ip_diferentes(data.get("remote_ip",""),data.get("meta.remote_ip","")),
                Labels.FEATURE_metaUser: self.get_meta_user(data.get("metaUser",None)),
                "label": data.get("label")
                })
        self.normalized_data = normalized_data

    def export(self, train,test):
        """TODO: Docstring for export.

        :train: TODO
        :test: TODO
        :returns: TODO

        """

        os.makedirs(self._base_path, exist_ok=True)
        train_path = os.path.join(self._base_path,"train.svm")
        test_path = os.path.join(self._base_path,"test.svm")

        with open(train_path,"w") as f:
            lines = []


            for d in train:
                lines.append("{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}\n".format(
                d["label"],
                Labels.FEATURE_IntervaloIP, d[Labels.FEATURE_IntervaloIP], #2
                Labels.FEATURE_formatInvalido, d[Labels.FEATURE_formatInvalido], #2
                Labels.FEATURE_uaInvalido, d[Labels.FEATURE_uaInvalido], #1
                Labels.FEATURE_typeMethod, d[Labels.FEATURE_typeMethod], #3
                Labels.FEATURE_cgnat, d[Labels.FEATURE_cgnat], #4
                Labels.FEATURE_ipsDiferentes, d[Labels.FEATURE_ipsDiferentes], #5
                Labels.FEATURE_metaUser, d[Labels.FEATURE_metaUser], #6
                        ))
            f.writelines(lines)
        with open(test_path,"w") as f:
            lines = []

            for data in test:
                lines.append("{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}\n".format(
                d["label"],
                Labels.FEATURE_IntervaloIP, d[Labels.FEATURE_IntervaloIP], #2
                Labels.FEATURE_formatInvalido, d[Labels.FEATURE_formatInvalido], #2
                Labels.FEATURE_uaInvalido, d[Labels.FEATURE_uaInvalido], #1
                Labels.FEATURE_typeMethod, d[Labels.FEATURE_typeMethod], #3
                Labels.FEATURE_cgnat, d[Labels.FEATURE_cgnat], #4
                Labels.FEATURE_ipsDiferentes, d[Labels.FEATURE_ipsDiferentes], #5
                Labels.FEATURE_metaUser, d[Labels.FEATURE_metaUser], #6
                   )
               )
            f.writelines(lines)

    def split(self, train):
        """TODO: Docstring for split.

        :train: TODO
        :test: TODO
        :returns: TODO

        """

        return self.normalized_data[:train],self.normalized_data[train:]


def handle_arguments():
    """TODO: Docstring for handle_arguments.
    :returns: TODO

    """
    parser = argparse.ArgumentParser("Transform json log into SVM vector")
    parser.add_argument("config",help="Config")

    return parser.parse_args()

class Classify(object):

    """Docstring for Classify. """

    def __init__(self,**config):
        """TODO: to be defined. """
        self._features=config.get("features",6)
        self._base_path=config.get("base_path")
        self._png_path=config.get("png_path","png")
        os.makedirs(self._png_path,exist_ok=True)


    def load_data(self,**kwargs):

        self.x, self.y = load_svmlight_file(self._base_path,n_features=6)
        X_train, X_test, y_train, y_test = train_test_split( self.x,self.y, test_size=0.2, random_state=1)
        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))


    def run(self):
        """TODO: Docstring for run.
        :returns: TODO

        """
        self.load_data()
        kf = KFold(n_splits=5,shuffle=True)

        count = 0

        for  train, test in kf.split(self.x):
            x_train = self.x[train]
            y_train = self.y[train]
            x_test = self.x[test]
            y_test = self.y[test]

            self.classify(x_train,y_train,x_test,y_test,SVC(),alg="SVC",parameters="",kfold=count)

            self.classify(x_train,y_train,x_test,y_test,KNeighborsClassifier(n_neighbors=2),alg="KNN",parameters="max_depth=2",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,KNeighborsClassifier(n_neighbors=3),alg="KNN",parameters="max_depth=3",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,KNeighborsClassifier(n_neighbors=5),alg="KNN",parameters="max_depth=5",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,KNeighborsClassifier(n_neighbors=7),alg="kNN",parameters="max_depth=7",kfold=count)

            self.classify(x_train,y_train,x_test,y_test,RandomForestClassifier(max_depth=2),alg="RandomForest",parameters="max_depth=2",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,RandomForestClassifier(max_depth=3),alg="RandomForest",parameters="max_depth=3",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,RandomForestClassifier(max_depth=5),alg="RandomForest",parameters="max_depth=5",kfold=count)
            self.classify(x_train,y_train,x_test,y_test,RandomForestClassifier(max_depth=7),alg="RandomForest",parameters="max_depth=7",kfold=count)
            #self.classify(x_train,y_train,x_test,y_test,SVC(),"SVC")
            #self.classify(x_train, y_train,x_test, y_test,Perceptron(tol=1e-3, random_state=0),"Perceptron")
            count += 1

    def classify(self,xtrain,ytrain,xtest,ytest,clf,**kwargs):

        clf.fit(xtrain, ytrain)
        y_pred = clf.predict(xtest)

        #print("Score: {}".format(clf.score(xtest, ytest)))
        #print(classification_report(ytest, y_pred, labels=[0,1],target_names=["Correto","Suspeito"]))
        cm = confusion_matrix(ytest, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correto","Suspeito"])
        disp.plot()
        f1 = f1_score(ytest,y_pred)
        recall = recall_score(ytest, y_pred)
        print("{:^6} & {:^8} & {:^1} & {:^6.4f} & {:^6.4f} & {:^6.4} \\\\".format(
            kwargs.get("alg"),
            kwargs.get("parameters"),
            kwargs.get("kfold"),
            clf.score(xtest,ytest),
            f1,recall
            ))
        plt.title('Matriz de confusão do {}'.format(kwargs.get("alg")))

        plt.savefig("pngs/matriz_confusao_{}_{}_{}.png".format(kwargs.get("alg"),kwargs.get("parameters"),kwargs.get("kfold")))

        fpr, tpr, _ = roc_curve(ytest, y_pred, pos_label=clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title('Curva roc para {}'.format(kwargs.get("alg")))
        plt.savefig("pngs/curva_roc_{}_{}_{}.png".format(kwargs.get("alg"),kwargs.get("parameters"),kwargs.get("kfold")))
    # Plot the projected points and show the evaluation score



def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = handle_arguments()
    with open(args.config,"r") as conf_f:
        config = yaml.load(conf_f,Loader=yaml.SafeLoader)

    #data = Data(**config["data"])
    #data.load()
    #data.select_attributes()
    #data.suffle()
    #data.rotule()
    #n = NormalizeData(data,"out_svm")
    #n.normalize()
    #train,test = n.split(8000)
    #n.export(train,test)

    c = Classify(**config["Classify"])
    c.run()
    #c.knn()
    #c.svc()


if __name__ == "__main__":
    main()
