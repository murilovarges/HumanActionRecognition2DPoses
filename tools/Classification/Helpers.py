import os
import smtplib
import cv2
import itertools
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib
import random
from PukKernel import PukKernel
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from sklearn.externals import joblib
from EnumTypes import ScalerType

class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

    def featuresVideoSIFT3D(self, descFile):
        descriptors = []
        with open(descFile) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(',')]
                descriptors.append(inner_list)
        d = np.asarray(descriptors, dtype='float32')
        return d

    def featuresVideoHARRIS3D(self, descFile):
        descriptors = []
        with open(descFile) as f:
            for line in f:
                if line[0] == '#':
                    continue
                inner_list = [elt.strip() for elt in line.split(' ')]
                data = inner_list[9:-1]
                descriptors.append(data)
        d = np.asarray(descriptors, dtype='float32')
        return d

    def featuresVideoC3D(self, descFile):
        descriptors = []
        with open(descFile) as f:
            for line in f:
                line = line.replace('[', '')
                line = line.replace(']',  '')
                inner_list = [elt.strip() for elt in line.split(',')]
                data = inner_list[:]
                descriptors.append(data)
        descr = np.asarray(descriptors, dtype='float32')
        label = descFile.split(os.sep)[-3]
        return descr, label

    def featuresVideoC3DConcat(self, descFile, base_path1, base_path2):
        descriptors = []
        with open(descFile) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(',')]
                data1 = inner_list[:]

        descFile = descFile.replace(base_path1, base_path2)
        with open(descFile) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(',')]
                data2 = inner_list[:]

        descriptors.append(data1 + data2)
        descr = np.asarray(descriptors, dtype='float32')
        label = descFile.split(os.sep)[-3]
        return descr, label


# def featuresVideoOld(self, descFile):
#	desc = np.loadtxt(descFile,dtype='float32',delimiter=',')
#	#d = np.asarray(descriptors)
#	return desc

class ClassifierHelpers:
    def __init__(self, classifier=LinearSVC(random_state=0), no_clusters=100):
        self.no_clusters = no_clusters
        self.kmeans_obj = KMeans(n_clusters=no_clusters, n_jobs=-1)
        self.descriptor_vstack = None
        self.descriptor_vstack_reduction = None
        self.features = None
        self.scale = None
        self.scaleCluster = None
        # self.clf = SVC(C=100, kernel='poly')
        # self.clf = SVC( kernel='rbf', C=100)
        # self.clf = SVC( kernel='rbf', C=2, gamma=0.0002)
        #self.clf = LinearSVC(random_state=0)
        #self.clf = SVC()
        self.clf = classifier
        self.PCA = PCA(n_components=30)

    def pca(self, descriptor_list, train=False):
        """
		pca 
		"""
        scaler = StandardScaler().fit(descriptor_list)
        descriptor_list = scaler.transform(descriptor_list);

        if train:
            self.PCA.fit(descriptor_list)

        descriptor_list = self.PCA.transform(descriptor_list)

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
		cluster using KMeans algorithm, 

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
                ret = self.kmeans_obj.predict(self.descriptor_vstack[count:count+l])
                for each in ret:
                    self.features[i][each] += 1

                #for j in range(l):
                #    idx = self.kmeans_obj.predict(self.descriptor_vstack[count + j].reshape(1, -1))
                #    self.features[i][idx] += 1
                count += l

                #ret = self.kmeans_obj.predict(descriptor_list[i])
                #for each in ret:
                #    self.features[i][each] += 1
            print("Vocabulary Histogram Generated")

    def formatND(self, l):
        """
		restructures list into vstack array of shape
		M samples x N features for sklearnreco

		"""
        self.descriptor_vstack = np.vstack(l)
        print("Number of Key Points: ", self.descriptor_vstack.shape)
        # vStack = np.array(l[0])
        # for remaining in l:
        #	vStack = np.vstack((vStack, remaining))
        # self.descriptor_vstack = vStack.copy()
        return self.descriptor_vstack

    def train(self, train_labels, train_model=None):
        """
		uses sklearn to train classifier
		"""
        print("Training Classifier")
        #print(self.clf)
        #print("Train labels", train_labels)
        self.clf.fit(self.features, train_labels)

        if train_model:
            joblib.dump(self.clf, train_model)
        print("Training completed")

    def loadModel(self, modelPath, loadScale=False):
        """
		"""
        print("Loading Classifier Model")
        self.clf = joblib.load(modelPath)
        if loadScale:
            self.scale = joblib.load(modelPath + '.scale')
        print("Loading completed")

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

    def getFilesFromList(self, path, list, number_dict=None, test=False, label_numeric=False):
        """
		- returns  a dictionary of all files 
		having key => value as  objectname => image path

		- returns total number of files.
		"""
        imlist = {}
        count = 0
        file = open(list, 'r')
        for line in file:
            line = line.replace('\n', '')
            if len(line) == 0:
                continue

            searchDir = os.path.join(path, line)
            if "vNonPorn" in line:
                if label_numeric:
                    label = number_dict['NonPorn']
                else:
                    label = 'NPornHard'

                if not os.path.exists(searchDir):
                    searchDir = os.path.join(path, 'vNonPornDifficulty', line)
                    if not os.path.exists(searchDir):
                        searchDir = os.path.join(path, 'vNonPornEasy', line)
                        if label_numeric:
                            label = number_dict['NonPorn']
                        else:
                            label = 'NPornEasy'

                        if not os.path.exists(searchDir):
                            continue
            else:
                if label_numeric:
                    label = number_dict['Porn']
                else:
                    label = 'Porn'
                if not os.path.exists(searchDir):
                    searchDir = os.path.join(path, 'vPorn', line)
                    if not os.path.exists(searchDir):
                        continue

            # if count > 200:
            #    break

            print(searchDir)
            video = searchDir.split(os.sep)[-1]
            print(" #### Reading Video ", video, " ##### ")
            imlist[video] = []
            c = 1
            for each in sorted(glob(searchDir + os.sep + "*")):
                # if c > 10:
                #    break
                print("Reading file ", each)
                if test:
                    imlist[video].append([each])
                else:
                    imlist[video].append([each, label])
                count += 1
                c += 1

        return [imlist, count]


    def getFilesFromDirectory(self, path, folders, number_dict=None, test=False, label_numeric=False):
        """
    		- returns  a dictionary of all files
    		having key => value as  objectname => image path

    		- returns total number of files.
    		"""
        img_list = {}
        count = 0

        for folder in folders:
            search_dir = os.path.join(path, folder)
            print(search_dir)
            if label_numeric:
                label = number_dict[folder]
            else:
                label = folder

            videos = sorted(glob(search_dir + '/*'))
            for video in videos:
                video_name = video.split(os.sep)[-1]
                print(" #### Reading Video ", video_name, " ##### ")
                img_list[video_name] = []
                c = 1
                for each in sorted(glob(video + os.sep + "*%s*" % self.file_filters)):
                    print("Reading file ", each)
                    if test:
                        img_list[video_name].append([each])
                    else:
                        img_list[video_name].append([each, label])
                    count += 1
                    c += 1

        return [img_list, count]


    def getFilesFromDirectoryLOOCV(self, path, datasets, folders, number_dict=None, test=False, label_numeric=False):
        """
    		- returns  a dictionary of all files
    		having key => value as  objectname => image path

    		- returns total number of files.
    		"""
        img_list = {}
        count = 0

        for dataset in datasets:
            for folder in folders:
                search_dir = os.path.join(path, dataset, folder)
                print(search_dir)
                if label_numeric:
                    label = number_dict[folder]
                else:
                    label = folder

                videos = sorted(glob(search_dir + '/*'))
                for video in videos:
                    video_name = video.split(os.sep)[-1]
                    print(" #### Reading Video ", video_name, " ##### ")
                    img_list[video_name] = []
                    c = 1
                    for each in sorted(glob(video + os.sep + "*%s*" % self.file_filters)):
                        print("Reading file ", each)
                        if test:
                            img_list[video_name].append([each])
                        else:
                            img_list[video_name].append([each, label])
                        count += 1
                        c += 1

        return [img_list, count]

    def getFilesFromDirectoryLOOCV_BOW(self, path, datasets,  extension='*.txt'):
        """
    		- returns  a dictionary of all files
    		having key => value as  objectname => image path

    		- returns total number of files.
    		"""
        img_list = []
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
                    img_list.append([featuresfile])
                    count += 1
                    #c += 1
                    #if c > 2:
                    #    break
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
                        img_list.append([featuresfile])
                        count += 1
                        #c += 1
                        #if c > 20:
                        #    break

        return [img_list, count]


    def getFiles(self, path, returnImage=False, extension='*.txt'):
        """
		- returns  a dictionary of all files 
		having key => value as  objectname => image path

		- returns total number of files.

		"""
        imlist = {}
        count = 0
        for each in sorted(glob(path + "*")):
            word = each.split(os.sep)[-1]
            print(" #### Reading category ", word, " ##### ")
            imlist[word] = []
            c = 1
            for imagefile in glob(path + word + os.sep + extension):
                # if c > 1:
                #	break
                print("Reading file ", imagefile)
                if returnImage:
                    im = cv2.imread(imagefile, 0)
                    imlist[word].append(im)
                else:
                    imlist[word].append(imagefile)

                count += 1
                c += 1

        return [imlist, count]


# def getFilesKeyPoints(self, path):
#	"""
#	- returns  a dictionary of all files
#	having key => value as  objectname => image path

#	- returns total number of files.

#	"""
#	imlist = {}
#	count = 0
#	for each in sorted(glob(path + "*")):
#		word = each.split(os.sep)[-1]
#		print( " #### Reading key points category ", word, " ##### ")
#		imlist[word] = []
#		c = 1
#		for keyfile in sorted(glob(path+word+os.sep+"*")):
#		    if c > 1:
#		        break

#		    print( "Reading file ", keyfile)
#		    imlist[word].append(keyfile)
#		    count +=1
#		    c +=1

#	return [imlist, count]


class MailHelpers:

    def __init__(self):
        self.fromaddr = 'mlrecogna@gmail.com'
        self.toaddrs = ['murilo.varges@gmail.com']
        self.msg = None
        self.username = 'mlrecogna@gmail.com'
        self.password = 'mlrecogna@2018'
        self.smtpserver = "smtp.gmail.com:587"

    def sendMail(self, subject, text, files=None):
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
