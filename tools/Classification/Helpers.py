import os
import smtplib
import itertools
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from sklearn.externals import joblib
from EnumTypes import ScalerType


class ClassifierHelpers:
    def __init__(self, classifier=LinearSVC(random_state=0), no_clusters=100):
        self.no_clusters = no_clusters
        self.kmeans_obj = KMeans(n_clusters=no_clusters, n_jobs=-1)
        self.descriptor_vstack = None
        self.descriptor_vstack_reduction = None
        self.features = None
        self.scale = None
        self.scaleCluster = None
        self.clf = classifier


    def normalize_cluster(self, scaler_type_cluster):
        """
		Normalize features before using KMeans algorithm
		"""
        if scaler_type_cluster.value == ScalerType.L2Normalize.value:
            print('Cluster - L2 Norm')
            self.descriptor_vstack = preprocessing.normalize(self.descriptor_vstack, norm="l2")
        elif scaler_type_cluster.value == ScalerType.StandardScaler.value:
            print('Cluster - StandardScaler')
            self.scaleCluster = preprocessing.StandardScaler().fit(self.descriptor_vstack)
            self.descriptor_vstack = self.scaleCluster.transform(self.descriptor_vstack)
        elif scaler_type_cluster.value == ScalerType.MinMaxScaler.value:
            print('Cluster - MinMaxScaler')
            self.scaleCluster = preprocessing.MinMaxScaler().fit(self.descriptor_vstack)
            self.descriptor_vstack = self.scaleCluster.transform(self.descriptor_vstack)

    def normalize(self, scaler_type):
        """
		Normalize features before using classifiers
		"""
        if scaler_type.value == ScalerType.L2Normalize.value:
            print('Feature Vector - L2 Norm')
            self.l2Normalize()
        elif scaler_type.value == ScalerType.StandardScaler.value:
            print('Feature Vector - StandardScaler')
            self.standardize()
        elif scaler_type.value == ScalerType.MinMaxScaler.value:
            print('Feature Vector - MinMaxScaler')
            self.minmax()

    def normalize_test_clusters(self, scaler_type_cluster, features):
        if scaler_type_cluster.value == ScalerType.L2Normalize.value:
            print('Cluster Test - L2 Norm')
            features = preprocessing.normalize(features, norm="l2")
        elif scaler_type_cluster.value == ScalerType.StandardScaler.value:
            print('Cluster Test - StandardScaler')
            features = self.scaleCluster.transform(features)
        elif scaler_type_cluster.value == ScalerType.MinMaxScaler.value:
            print('Cluster Test - MinMaxScaler')
            features = self.scaleCluster.transform(features)

        return features

    def normalize_test(self, scaler_type, features):
        """
		Normalize features before using classifiers
		"""
        if scaler_type.value == ScalerType.L2Normalize.value:
            print('Feature Vector Test - L2 Norm')
            features = preprocessing.normalize(features, norm="l2")
        elif scaler_type.value == ScalerType.StandardScaler.value:
            print('Feature Vector Test - StandardScaler')
            features = self.scale.transform(features)
        elif scaler_type.value == ScalerType.MinMaxScaler.value:
            print('Feature Vector Test - MinMaxScaler')
            features = self.scale.transform(features)

        return features

    def standardize(self, std=None, train_model=None):
        """
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
		"""
        if std is None:
            self.scale = preprocessing.StandardScaler().fit(self.features)
            self.features = self.scale.transform(self.features)
        else:
            print("STD not none. External STD supplied")
            self.features = std.transform(self.features)
        if train_model is not None:
            joblib.dump(self.scale, train_model + '.scale')

    def minmax(self, std=None, train_model=None):
        """
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
		"""
        if std is None:
            self.scale = preprocessing.MinMaxScaler().fit(self.features)
            self.features = self.scale.transform(self.features)
        else:
            print("STD not none. External STD supplied")
            self.features = std.transform(self.features)
        if train_model is not None:
            joblib.dump(self.scale, train_model + '.scale')

    def l2Normalize(self):
        """		
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
        """
        self.features = preprocessing.normalize(self.features, norm="l2")

    def cluster(self, no_samples=None):
        """
		cluster using KMeans algorithm
		"""
        if no_samples:
            self.descriptor_vstack_reduction = self.descriptor_vstack[np.random.choice(self.descriptor_vstack.shape[0],
                                                                                       size=no_samples, replace=False)]
            print("Cluster Number of Key Points: ", self.descriptor_vstack_reduction.shape)
            self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack_reduction)
        else:
            self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

        print("Kmeans clusters Generated")

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):

        """
		Each cluster denotes a particular visual word 
		Every image can be represeted as a combination of multiple 
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word 

		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images

		"""

        self.features = np.array([np.zeros(self.no_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.features[i][idx] += 1
            old_count += l
        print("Vocabulary Histogram Generated")

    def developVocabulary_samples(self, n_images, descriptor_list):

        """
    		Each cluster denotes a particular visual word
    		Every image can be represeted as a combination of multiple
    		visual words. The best method is to generate a sparse histogram
    		that contains the frequency of occurence of each visual word

    		Thus the vocabulary comprises of a set of histograms of encompassing
    		all descriptions for all images

    		"""

        self.features = np.array([np.zeros(self.no_clusters) for i in range(n_images)])
        count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            ret = self.kmeans_obj.predict(self.descriptor_vstack[count:count + l])
            for each in ret:
                self.features[i][each] += 1

            # for j in range(l):
            #    idx = self.kmeans_obj.predict(self.descriptor_vstack[count + j].reshape(1, -1))
            #    self.features[i][idx] += 1
            count += l

            # ret = self.kmeans_obj.predict(descriptor_list[i])
            # for each in ret:
            #    self.features[i][each] += 1
        print("Vocabulary Histogram Generated")

    def formatND(self, l):
        """
		restructures list into vstack array of shape
		M samples x N features for sklearnreco
		"""
        self.descriptor_vstack = np.vstack(l)
        print("Number of Key Points: ", self.descriptor_vstack.shape)
        return self.descriptor_vstack

    def train(self, train_labels, train_model=None):
        """
		uses sklearn to train classifier
		"""
        print("Training Classifier")
        self.clf.fit(self.features, train_labels)

        if train_model:
            joblib.dump(self.clf, train_model)
        print("Training completed")

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.features

        x_scalar = np.arange(self.no_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.no_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              # cmap=plt.cm.Blues,
                              cmap=plt.cm.coolwarm,
                              show=True,
                              fileName='picture.png'):
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
        # plt.xticks(tick_marks, classes, fontsize=3.5, rotation=90)
        # plt.yticks(tick_marks, classes, fontsize=3.5)
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
        plt.savefig(fileName, bbox_inches='tight', dpi=600)


class FileHelpers:

    def __init__(self, fl_filters=None):
        self.file_filters = fl_filters
        pass

    def getLabelsFromFile(self, pathLabels):
        name_dict = {}
        number_dict = {}
        label_count = 0
        file = open(pathLabels, 'r')
        for word in file:
            word = word.replace('\n', '')
            name_dict[str(label_count)] = word
            number_dict[str(word)] = label_count
            label_count += 1

        return [name_dict, number_dict, label_count]

    def getFilesFromDirectory(self, path, datasets, extension='*.txt'):
        """
    		- returns  a list of features files
    	"""
        files_list = []
        count = 0
        if datasets is None:
            search_dir = path
            print(search_dir)

            for each in sorted(glob(search_dir + "/*")):
                word = each.split(os.sep)[-1]
                print(" #### Reading category ", word, " ##### ")
                c = 1
                for featuresfile in glob(each + os.sep + extension):
                    print("Reading file ", featuresfile)
                    files_list.append([featuresfile])
                    count += 1
        else:
            for dataset in datasets:
                search_dir = os.path.join(path, dataset)
                print(search_dir)

                for each in sorted(glob(search_dir + "/*")):
                    word = each.split(os.sep)[-1]
                    print(" #### Reading category ", word, " ##### ")
                    c = 1
                    for featuresfile in sorted(glob(each + os.sep + extension)):
                        print("Reading file ", featuresfile)
                        files_list.append([featuresfile])
                        count += 1

        return [files_list, count]

    def formatFeatures(self, descFile):
        descriptors = []
        with open(descFile) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(',')]
                descriptors.append(inner_list)
        d = np.asarray(descriptors, dtype='float32')
        return d

class MailHelpers:

    def __init__(self):
        self.configured = False
        self.fromaddr = 'from_email_address@gmail.com'
        self.toaddrs = ['to_email_address@gmail.com']
        self.msg = None
        self.username = 'user@gmail.com'
        self.password = 'password'
        self.smtpserver = "smtp.gmail.com:587"

    def sendMail(self, subject, text, files=None):
        if self.configured:
            # assert isinstance(self.toaddrs, list)
            msg = MIMEMultipart()
            msg['From'] = self.fromaddr
            msg['To'] = COMMASPACE.join(self.toaddrs)
            msg['Date'] = formatdate(localtime=True)
            msg['Subject'] = subject

            msg.attach(MIMEText(text))

            for f in files or []:
                with open(f, "rb") as fil:
                    part = MIMEApplication(
                        fil.read(),
                        Name=basename(f)
                    )
                # After the file is closed
                part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
                msg.attach(part)

            smtp = smtplib.SMTP(self.smtpserver)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.sendmail(self.fromaddr, self.toaddrs, msg.as_string())
            smtp.close()
