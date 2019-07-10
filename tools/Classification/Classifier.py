import errno
import csv
import datetime
import errno
import os
import numpy as np
from EnumTypes import ScalerType
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GMM
from FisherVector import power_normalize, L2_normalize, fisher_vector
from Helpers import ImageHelpers, ClassifierHelpers, FileHelpers, MailHelpers
from PukKernel import PukKernel


class Classifier:
    def __init__(self,
                 classifier=LinearSVC(random_state=0),
                 fl_filters=None,
                 results_file=None,
                 no_clusters=100,
                 no_samples=800):
        self.no_clusters = no_clusters
        self.no_samples = no_samples
        self.OsName = None
        self.scaler_type = ScalerType.StandardScaler
        self.scaler_type_cluster = ScalerType.StandardScaler
        self.test_name = None
        self.base_path = None
        self.base_path2 = None
        self.base_path3 = None
        self.label_path = None
        self.train_path = None
        self.test_path = None
        self.datasets = None
        self.folders = None
        self.parameters = None
        self.aggregateVideoFeatures = True
        self.im_helper = ImageHelpers()
        self.classifier_helper = ClassifierHelpers(classifier=classifier, no_clusters=no_clusters)
        self.file_helper = FileHelpers(fl_filters)
        self.mail_helper = MailHelpers()
        self.descr_files = None
        self.trainVideoCount = 0
        self.train_labels = np.array([])
        self.groups = np.array([])
        self.name_dict = {}
        self.number_dict = {}
        self.count_class = 0
        self.descriptor_list = []
        self.results_file = results_file

    def trainTestModelCV(self):
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)
        self.allVideos, self.allVideosCount = self.file_helper.getFilesFromDirectoryLOOCV(self.base_path, self.datasets,
                                                                                          self.folders,
                                                                                          self.number_dict, test=False,
                                                                                          label_numeric=True)

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf
        ctd_groups = 0
        # extract Features from each image/video
        for video, descrFileList in sorted(self.allVideos.items()):
            ctd_groups += 1
            print("processing video: ", video)
            descriptor_video = []
            for descrFile in descrFileList:
                file = descrFile[0]
                labelInt = int(descrFile[1])
                print("Computing Features for ", file)
                descriptor, label = self.im_helper.featuresVideoC3D(file)
                if self.aggregateVideoFeatures:
                    descriptor_video.append(descriptor)
                else:
                    self.train_labels = np.append(self.train_labels, labelInt)
                    self.descriptor_list.append(descriptor)
                    self.groups = np.append(self.groups, ctd_groups)

            if self.aggregateVideoFeatures:
                self.train_labels = np.append(self.train_labels, labelInt)
                descriptor_video_mean = np.mean(descriptor_video, axis=0)
                self.descriptor_list.append(descriptor_video_mean)
                self.groups = np.append(self.groups, ctd_groups)

        # preparing Data
        print("Preparing Data")
        self.classifier_helper.features = self.classifier_helper.formatND(self.descriptor_list)
        print("Preparing Data done!")

        # loocv = LeaveOneOut()
        loocv = LeaveOneGroupOut()

        clf = make_pipeline(preprocessing.StandardScaler(), decomposition.PCA(n_components=100),
                            self.classifier_helper.clf)

        # predictions = cross_val_predict(clf, self.classifier_helper.features, self.train_labels, cv=loocv, n_jobs=-1)
        predictions = cross_val_predict(clf,
                                        self.classifier_helper.features,
                                        self.train_labels,
                                        groups=self.groups,
                                        cv=loocv,
                                        verbose=1,
                                        n_jobs=15)
        # pre_dispatch=10)
        # n_jobs=10)
        p_video = []
        l_video = []
        for i in range(1, ctd_groups + 1):
            idx = np.where(self.groups == i)
            p = predictions[idx]
            l = self.train_labels[idx]
            counts = np.bincount(p.astype(int))
            p_label = np.argmax(counts)
            p_video.append(float(p_label))
            l_video.append(l[0])
            # print(p)

        if self.aggregateVideoFeatures:
            videosCount = predictions.size
            self.saveResults(None, predictions, self.train_labels, videosCount)
        else:
            videosCount = ctd_groups
            clipsCount = predictions.size
            self.saveResults(None, p_video, l_video, videosCount, None, predictions, self.train_labels, clipsCount)

    def trainModel(self, train_model=None, use_file_list=True):
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)
        if train_model is None or not os.path.exists(train_model):
            # read file. prepare file lists.
            if use_file_list:
                self.trainVideos, self.trainVideoCount = self.file_helper.getFilesFromList \
                    (self.base_path, self.train_path, self.number_dict, test=False, label_numeric=True)
            else:
                self.trainVideos, self.trainVideoCount = self.file_helper.getFilesFromDirectory \
                    (self.train_path, self.folders, self.number_dict, test=False, label_numeric=True)

            # extract Features from each image/video
            for video, descrFileList in sorted(self.trainVideos.items()):
                print("processing video: ", video)
                descriptor_video = []
                for descrFile in descrFileList:
                    file = descrFile[0]
                    labelInt = int(descrFile[1])
                    print("Computing Features for ", file)
                    if self.base_path2:
                        descriptor, label = self.im_helper.featuresVideoC3DConcat(file, self.base_path, self.base_path2)
                    else:
                        descriptor, label = self.im_helper.featuresVideoC3D(file)
                    if self.aggregateVideoFeatures:
                        descriptor_video.append(descriptor)
                    else:
                        self.train_labels = np.append(self.train_labels, labelInt)
                        self.descriptor_list.append(descriptor)

                if self.aggregateVideoFeatures:
                    self.train_labels = np.append(self.train_labels, labelInt)
                    descriptor_video_mean = np.mean(descriptor_video, axis=0)
                    self.descriptor_list.append(descriptor_video_mean)

            # preparing Data
            print("Preparing Data")
            self.classifier_helper.features = self.classifier_helper.formatND(self.descriptor_list)
            print("Preparing Data done!")

            if self.scaler_type == ScalerType.L2Normalize:
                self.classifier_helper.normalizeAll()
            elif self.scaler_type == ScalerType.StandardScaler:
                self.classifier_helper.standardize(std=None, train_model=train_model)

            self.classifier_helper.train(self.train_labels, train_model)
        else:
            loadScale = self.scaler_type == ScalerType.StandardScaler
            self.classifier_helper.loadModel(train_model, loadScale)

        current_date = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        msg = "Training Done at: {}".format(current_date)
        self.mail_helper.sendMail("Training Done: %s - %s" % (self.test_name, self.OsName), msg, None)

        self.parameters += 'SVM Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

    def trainModelBoVW(self):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)
        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path, False, '*_full.json')

        self.parameters += 'SVM Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in sorted(self.images.items()):
            self.name_dict[str(label_count)] = word
            print("Computing Features for ", word)
            for descFile in imlist:
                self.train_labels = np.append(self.train_labels, label_count)
                des = self.im_helper.featuresVideoSIFT3D(descFile)
                self.descriptor_list.append(des)

            label_count += 1

        # format data as nd array
        self.classifier_helper.formatND(self.descriptor_list)
        # perform features normalize before clustering
        self.classifier_helper.normalize_cluster(self.scaler_type_cluster)
        # perform clustering
        self.classifier_helper.cluster()
        # build vocabulary
        self.classifier_helper.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        # self.classifier_helper.plotHist()

        # perform normalization in the histogram (vocabulary)
        self.classifier_helper.normalize(self.scaler_type)

        self.classifier_helper.train(self.train_labels)

    def trainModelBoVW_LOOCV(self, extension='*.txt'):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path,
                                                                                            self.datasets,
                                                                                            extension)

        self.parameters += 'SVM Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd = np.asarray(self.images)
        loo = LeaveOneOut()

        predictions = []
        p = []
        l = []
        c = 0
        hits = 0
        for train, test in loo.split(features_nd):
            feature_test = str(features_nd[test][0][0])
            class_name_test = feature_test.split(os.sep)[-2]
            c += 1
            currenInvDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('Step: %i/%i - %s - %s' % (c, features_nd.shape[0], currenInvDate, feature_test))
            self.descriptor_list = []
            self.train_labels = []
            for feature in features_nd[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                # print(feature)
                des = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list.append(des)

            # format data as nd array
            self.classifier_helper.formatND(self.descriptor_list)
            # perform features normalize before clustering
            self.classifier_helper.normalize_cluster(self.scaler_type_cluster)

            # perform clustering
            if self.no_samples is not None:
                self.classifier_helper.cluster(no_samples=self.no_samples)
                self.classifier_helper.developVocabulary_samples(n_images=train.shape[0],
                                                                 descriptor_list=self.descriptor_list)
            else:
                self.classifier_helper.cluster()
                self.classifier_helper.developVocabulary(n_images=train.shape[0],
                                                         descriptor_list=self.descriptor_list)

            # perform normalization in the histogram (vocabulary)
            self.classifier_helper.normalize(self.scaler_type)

            # train classifier
            self.classifier_helper.train(self.train_labels)

            # recognize test sample
            cl = self.recognizeBoVW(feature_test)
            # predicted label
            p.extend([cl])
            class_name_predict = self.name_dict[str(cl)]

            if class_name_test == class_name_predict:
                hits += 1

            msg_progress = 'Hits: %i/%i  -  Accuracy: %.4f\n\n' % (hits, c, hits / c)
            print(msg_progress)

            if c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), msg_progress)

            # real label
            l.extend([self.number_dict[feature_test.split(os.sep)[-2]]])
            predictions.append({
                'image': feature_test,
                'class': cl,
                'object_name': self.name_dict[str(cl)]
            })

        self.saveResults(predictions, p, l, features_nd.shape[0])

    def FV_LOOCV_Features(self, extension='*.*'):
        """
        This method contains the entire module
        to compute features for Visualization plot.
        """
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.images1, self.trainImageCount1 = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path,
                                                                                              self.datasets,
                                                                                              extension)
        self.images2, self.trainImageCount2 = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path2,
                                                                                              self.datasets,
                                                                                              extension)

        features_nd1 = np.asarray(self.images1)
        features_nd2 = np.asarray(self.images2)
        features_nd1.sort(axis=0)
        features_nd2.sort(axis=0)

        labels_train = []
        self.descriptor_list1 = []
        self.descriptor_list2 = []

        for feature in features_nd1:
            feature = feature[0]
            label_number = self.number_dict[feature.split(os.sep)[-2]]
            label_name = self.name_dict[str(label_number)]
            labels_train = np.append(labels_train, label_name)
            des1 = self.im_helper.featuresVideoSIFT3D(feature)
            self.descriptor_list1.append(des1)

        for feature in features_nd2:
            feature = feature[0]
            des2 = self.im_helper.featuresVideoSIFT3D(feature)
            self.descriptor_list2.append(des2)

        # format data as nd array
        ft1 = self.classifier_helper.formatND(self.descriptor_list1)
        ft2 = self.classifier_helper.formatND(self.descriptor_list2)

        gmm1 = GMM(n_components=self.no_clusters, covariance_type='diag', verbose=0)
        gmm1.fit(ft1)

        gmm2 = GMM(n_components=self.no_clusters, covariance_type='diag', verbose=0)
        gmm2.fit(ft2)

        fv_dim1 = self.no_clusters + 2 * self.no_clusters * ft1.shape[1]
        fv_dim2 = self.no_clusters + 2 * self.no_clusters * ft2.shape[1]
        print(fv_dim1, fv_dim2)
        n_videos = features_nd1.shape[0]
        features1 = np.array([np.zeros(fv_dim1) for i in range(n_videos)])
        features2 = np.array([np.zeros(fv_dim2) for i in range(n_videos)])
        count1 = 0
        count2 = 0
        for i in range(n_videos):
            len_video1 = len(self.descriptor_list1[i])
            fv1 = fisher_vector(ft1[count1:count1 + len_video1], gmm1)
            features1[i] = fv1
            count1 += len_video1

            len_video2 = len(self.descriptor_list2[i])
            fv2 = fisher_vector(ft2[count2:count2 + len_video2], gmm2)
            features2[i] = fv2
            count2 += len_video2

        print(features1.shape)
        print('Data normalization. 1')
        scaler1 = StandardScaler()
        # train normalization
        features1 = scaler1.fit_transform(features1)
        features1 = power_normalize(features1, 0.5)
        features1 = L2_normalize(features1)

        print(features2.shape)
        print('Data normalization. 2')
        scaler2 = StandardScaler()
        # train normalization
        features2 = scaler2.fit_transform(features2)
        features2 = power_normalize(features2, 0.5)
        features2 = L2_normalize(features2)

        ## concatenate two fv train
        features_train = np.concatenate((features1, features2), axis=1)

        return features_train, labels_train

    def trainModelFV_LOOCV_Fusion(self, extension='*.*'):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.images1, self.trainImageCount1 = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path,
                                                                                              self.datasets,
                                                                                              extension)
        self.images2, self.trainImageCount2 = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path2,
                                                                                              self.datasets,
                                                                                              extension)
        '''
        self.images3, self.trainImageCount3 = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path3,
                                                                                            self.datasets,
                                                                                            extension)
        '''

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd1 = np.asarray(self.images1)
        features_nd2 = np.asarray(self.images2)
        # features_nd3 = np.asarray(self.images3)
        features_nd1.sort(axis=0)
        features_nd2.sort(axis=0)
        # features_nd3.sort(axis=0)
        loo = LeaveOneOut()
        predictions = []
        pre = []
        lab = []
        hits = 0
        c = 0
        for train, test in loo.split(features_nd1):
            feature_test_file1 = str(features_nd1[test][0][0])
            feature_test_file2 = str(features_nd2[test][0][0])
            # feature_test_file3 = str(features_nd3[test][0][0])
            class_name_test = feature_test_file1.split(os.sep)[-2]
            c += 1

            currenInvDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('Step: %i/%i - %s\n%s\n%s' % (c, features_nd1.shape[0], currenInvDate,
                                                feature_test_file1, feature_test_file2))
            # if c == 1 or c % 25 == 0:
            #    self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), "Samples processed: %i" % c)

            self.descriptor_list1 = []
            self.descriptor_list2 = []
            self.descriptor_list3 = []
            self.train_labels = []
            for feature in features_nd1[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                des1 = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list1.append(des1)

            for feature in features_nd2[train]:
                feature = feature[0]
                des2 = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list2.append(des2)
            '''
            for feature in features_nd3[train]:
                feature = feature[0]
                des3 = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list3.append(des3)
            '''
            # format data as nd array
            ft1 = self.classifier_helper.formatND(self.descriptor_list1)
            ft2 = self.classifier_helper.formatND(self.descriptor_list2)
            # ft3 = self.classifier_helper.formatND(self.descriptor_list3)

            gmm1 = GMM(n_components=self.no_clusters, covariance_type='diag', verbose=0)
            gmm1.fit(ft1)

            gmm2 = GMM(n_components=self.no_clusters, covariance_type='diag', verbose=0)
            gmm2.fit(ft2)

            # gmm3 = GMM(n_components=self.no_clusters, covariance_type='diag', verbose=0)
            # gmm3.fit(ft3)

            fv_dim1 = self.no_clusters + 2 * self.no_clusters * ft1.shape[1]
            fv_dim2 = self.no_clusters + 2 * self.no_clusters * ft2.shape[1]
            # fv_dim3 = self.no_clusters + 2 * self.no_clusters * ft3.shape[1]
            print(fv_dim1, fv_dim2)
            n_videos = train.shape[0]
            features1 = np.array([np.zeros(fv_dim1) for i in range(n_videos)])
            features2 = np.array([np.zeros(fv_dim2) for i in range(n_videos)])
            # features3 = np.array([np.zeros(fv_dim3) for i in range(n_videos)])
            count1 = 0
            count2 = 0
            # count3 = 0
            for i in range(n_videos):
                len_video1 = len(self.descriptor_list1[i])
                fv1 = fisher_vector(ft1[count1:count1 + len_video1], gmm1)
                features1[i] = fv1
                count1 += len_video1

                len_video2 = len(self.descriptor_list2[i])
                fv2 = fisher_vector(ft2[count2:count2 + len_video2], gmm2)
                features2[i] = fv2
                count2 += len_video2

                '''
                len_video3 = len(self.descriptor_list3[i])
                fv3 = fisher_vector(ft3[count3:count3+len_video3], gmm3)
                features3[i] = fv3
                count3 += len_video3
                '''

            print(features1.shape)
            print('Data normalization. 1')
            scaler1 = StandardScaler()
            # train normalization
            features1 = scaler1.fit_transform(features1)
            features1 = power_normalize(features1, 0.5)
            features1 = L2_normalize(features1)

            print(features2.shape)
            print('Data normalization. 2')
            scaler2 = StandardScaler()
            # train normalization
            features2 = scaler2.fit_transform(features2)
            features2 = power_normalize(features2, 0.5)
            features2 = L2_normalize(features2)

            '''
            print(features3.shape)
            print('Data normalization. 3')
            scaler3 = StandardScaler()
            # train normalization
            features3 = scaler3.fit_transform(features3)
            features3 = power_normalize(features3, 0.5)
            features3 = L2_normalize(features3)
            '''

            # real label
            lab.extend([self.number_dict[feature_test_file1.split(os.sep)[-2]]])

            # test features 1
            feature_test1 = self.im_helper.featuresVideoSIFT3D(feature_test_file1)
            test_fv1 = fisher_vector(feature_test1, gmm1)
            # train normalization
            test_fv1 = test_fv1.reshape(1, -1)
            test_fv1 = scaler1.transform(test_fv1)
            test_fv1 = power_normalize(test_fv1, 0.5)
            test_fv1 = L2_normalize(test_fv1)

            # test features 2
            feature_test2 = self.im_helper.featuresVideoSIFT3D(feature_test_file2)
            test_fv2 = fisher_vector(feature_test2, gmm2)
            # train normalization
            test_fv2 = test_fv2.reshape(1, -1)
            test_fv2 = scaler2.transform(test_fv2)
            test_fv2 = power_normalize(test_fv2, 0.5)
            test_fv2 = L2_normalize(test_fv2)

            '''
            # test features 3
            feature_test3 = self.im_helper.featuresVideoSIFT3D(feature_test_file3)
            test_fv3 = fisher_vector(feature_test3, gmm3)
            # train normalization
            test_fv3 = test_fv3.reshape(1, -1)
            test_fv3 = scaler3.transform(test_fv3)
            test_fv3 = power_normalize(test_fv3, 0.5)
            test_fv3 = L2_normalize(test_fv3)
            '''

            ## concatenate two fv test
            feature_test = np.concatenate((test_fv1, test_fv2), axis=1).reshape(1, -1)

            ## concatenate two fv train
            feature_train = np.concatenate((features1, features2), axis=1)

            # train classifiers
            self.classifier_helper.clf.fit(feature_train, self.train_labels)
            cl = int(self.classifier_helper.clf.predict(feature_test)[0])
            class_name_predict = self.name_dict[str(cl)]
            if class_name_test == class_name_predict:
                hits += 1

            msg_progress = 'Hits: %i/%i  -  Accuracy: %.4f\n\n' % (hits, c, hits / c)
            print(msg_progress)
            if c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), msg_progress)

            # predicted label
            pre.extend([cl])
            predictions.append({
                'image1': feature_test_file1,
                'image2': feature_test_file2,
                'class': cl,
                'object_name': self.name_dict[str(cl)]
            })

        self.saveResults(predictions, pre, lab, features_nd1.shape[0])

    def trainModelFV_LOOCV_Classifiers(self, extension='*.txt'):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """

        names = [  # "Nearest Neighbors",
            "Linear SVM"]
        # "RBF SVM",
        # "Poly SVM",
        # "Puk SVM",
        # "Gaussian Process",
        # "Decision Tree",
        # "Random Forest",
        # "Neural Net"]
        # "AdaBoost",
        # "Naive Bayes",
        # "QDA"]

        classifiers = [
            # KNeighborsClassifier(3),
            SVC(kernel='linear')]
        # SVC(kernel='rbf'),
        # SVC(kernel='poly'),
        # SVC(kernel=PukKernel),
        # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1)]
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()]

        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path,
                                                                                            self.datasets,
                                                                                            extension)

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd = np.asarray(self.images)
        loo = LeaveOneOut()
        predictions = {}
        p = {}
        l = []
        hits = {}
        for name in names:
            predictions[name] = []
            p[name] = []
            hits[name] = 0

        c = 0
        for train, test in loo.split(features_nd):
            feature_test_file = str(features_nd[test][0][0])
            class_name_test = feature_test_file.split(os.sep)[-2]
            c += 1
            currenInvDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('Step: %i/%i - %s - %s' % (c, features_nd.shape[0], currenInvDate, feature_test_file))
            # if c == 1 or c % 25 == 0:
            #    self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), "Samples processed: %i" % c)

            self.descriptor_list = []
            self.train_labels = []
            for feature in features_nd[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                des = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list.append(des)

            # format data as nd array
            self.classifier_helper.formatND(self.descriptor_list)

            gmm = GMM(n_components=self.no_clusters, covariance_type='diag')
            gmm.fit(self.classifier_helper.descriptor_vstack)

            fv_dim = self.no_clusters + 2 * self.no_clusters * self.classifier_helper.descriptor_vstack.shape[1]
            print(fv_dim)
            n_videos = train.shape[0]
            features = np.array([np.zeros(fv_dim) for i in range(n_videos)])
            count = 0
            for i in range(n_videos):
                len_video = len(self.descriptor_list[i])
                fv = fisher_vector(self.classifier_helper.descriptor_vstack[count:count + len_video], gmm)
                features[i] = fv
                count += len_video

            print(features.shape)
            print('Data normalization.')
            scaler = StandardScaler()
            # train normalization
            features = scaler.fit_transform(features)
            features = power_normalize(features, 0.5)
            features = L2_normalize(features)

            # real label
            l.extend([self.number_dict[feature_test_file.split(os.sep)[-2]]])

            # test features
            feature_test = self.im_helper.featuresVideoSIFT3D(feature_test_file)
            test_fv = fisher_vector(feature_test, gmm)
            # train normalization
            test_fv = test_fv.reshape(1, -1)
            test_fv = scaler.transform(test_fv)
            test_fv = power_normalize(test_fv, 0.5)
            test_fv = L2_normalize(test_fv)

            # train classifiers
            for name, clf in zip(names, classifiers):
                print(name)
                clf.fit(features, self.train_labels)
                cl = int(clf.predict(test_fv)[0])
                class_name_predict = self.name_dict[str(cl)]
                if class_name_test == class_name_predict:
                    hits[name] += 1

                # predicted label
                p[name].extend([cl])
                predictions[name].append({
                    'image': feature_test_file,
                    'class': cl,
                    'object_name': self.name_dict[str(cl)]
                })
            msg_progress = ''
            for name in names:
                msg_progress += 'Classifier: %s - Hits:%i/%i - Accuracy: %.4f\n' % (
                    name.ljust(20), hits[name], c, hits[name] / c)

            print(msg_progress)
            print('\n\n')
            if c == 1 or c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), msg_progress)

        for name in names:
            print(name)
            self.saveResults(predictions[name], p[name], l, features_nd.shape[0], classifier_name=name)

    def trainModelBoVW_LOOCV_Classifiers(self, extension='*.txt'):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """

        names = ["Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 # "Poly SVM",
                 "Puk SVM",
                 # "Gaussian Process",
                 "Decision Tree",
                 # "Random Forest",
                 "Neural Net"]
        # "AdaBoost"
        # "Naive Bayes"
        # "QDA"]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel='linear'),
            SVC(kernel='rbf'),
            # SVC(kernel='poly', C=10),
            SVC(kernel=PukKernel),
            # GaussianProcessClassifier(1.0 * RBF(1.0),n_jobs=-1),
            DecisionTreeClassifier(max_depth=5),
            # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1)]
        # AdaBoostClassifier(),
        # GaussianNB()]
        # QuadraticDiscriminantAnalysis()]

        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFilesFromDirectoryLOOCV_BOW(self.base_path,
                                                                                            self.datasets,
                                                                                            extension)

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd = np.asarray(self.images)
        loo = LeaveOneOut()
        predictions = {}
        p = {}
        l = []
        hits = {}
        for name in names:
            predictions[name] = []
            p[name] = []
            hits[name] = 0

        c = 0
        for train, test in loo.split(features_nd):
            feature_test = str(features_nd[test][0][0])
            class_name_test = feature_test.split(os.sep)[-2]
            c += 1
            currenInvDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('Step: %i/%i - %s - %s' % (c, features_nd.shape[0], currenInvDate, feature_test))
            # if c == 1 or c % 25 == 0:
            #    self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), "Samples processed: %i" % c)

            self.descriptor_list = []
            self.train_labels = []
            for feature in features_nd[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                des = self.im_helper.featuresVideoSIFT3D(feature)
                self.descriptor_list.append(des)

            # format data as nd array
            self.classifier_helper.formatND(self.descriptor_list)
            # perform features normalize before clustering
            self.classifier_helper.normalize_cluster(self.scaler_type_cluster)

            # perform clustering
            if self.no_samples is not None:
                self.classifier_helper.cluster(no_samples=self.no_samples)
                self.classifier_helper.developVocabulary_samples(n_images=train.shape[0],
                                                                 descriptor_list=self.descriptor_list)
            else:
                self.classifier_helper.cluster()
                self.classifier_helper.developVocabulary(n_images=train.shape[0],
                                                         descriptor_list=self.descriptor_list)
            # real label
            l.extend([self.number_dict[feature_test.split(os.sep)[-2]]])

            # perform normalization in the histogram (vocabulary)
            self.classifier_helper.normalize(self.scaler_type)

            # train classifiers
            # self.classifier_helper.train(self.train_labels)
            for name, clf in zip(names, classifiers):
                print(name)
                clf.fit(self.classifier_helper.features, self.train_labels)
                vocab = self.prepareVocab(feature_test)
                cl = int(clf.predict(vocab)[0])

                class_name_predict = self.name_dict[str(cl)]

                if class_name_test == class_name_predict:
                    hits[name] += 1

                # predicted label
                p[name].extend([cl])
                predictions[name].append({
                    'image': feature_test,
                    'class': cl,
                    'object_name': self.name_dict[str(cl)]
                })
            msg_progress = ''
            for name in names:
                msg_progress += 'Classifier: %s - Hits:%i/%i - Accuracy: %.4f\n' % (
                name.ljust(20), hits[name], c, hits[name] / c)

            print(msg_progress)
            print('\n\n')
            if c == 1 or c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName), msg_progress)

        for name in names:
            print(name)
            self.saveResults(predictions[name], p[name], l, features_nd.shape[0], classifier_name=name)

    def recognizeClip(self, test_img):
        des, lab = self.im_helper.featuresVideoC3D(test_img)
        des = self.classifier_helper.formatND(des)

        # Scale the features
        des = des.reshape(1, -1)
        if self.scaler_type == ScalerType.L2Normalize:
            des = self.classifier_helper.normalize(des)
        elif self.scaler_type == ScalerType.StandardScaler:
            des = self.classifier_helper.scale.transform(des)

        # predict the class of the image
        lb = self.classifier_helper.clf.predict(des)
        print("Image belongs to class : ", self.name_dict[str(int(lb[0]))])
        return int(lb[0])

    def recognizeVideo(self, video, test_files):
        clipsCount = 0
        descriptor_video = []
        for descrFile in test_files:
            clipsCount = clipsCount + 1
            file = descrFile[0]
            print("processing clip", file)
            if self.base_path2:
                des, lab = self.im_helper.featuresVideoC3DConcat(file, self.base_path, self.base_path2)
            else:
                des, lab = self.im_helper.featuresVideoC3D(file)

            descriptor_video.append(des)

        descriptor_video_mean = np.mean(descriptor_video, axis=0)
        descriptor_video_mean = self.classifier_helper.formatND(descriptor_video_mean)

        # Scale the features
        descriptor_video_mean = descriptor_video_mean.reshape(1, -1)
        if self.scaler_type == ScalerType.L2Normalize:
            descriptor_video_mean = self.classifier_helper.normalize(descriptor_video_mean)
        elif self.scaler_type == ScalerType.StandardScaler:
            descriptor_video_mean = self.classifier_helper.scale.transform(descriptor_video_mean)

        # predict the class of the image
        lb = self.classifier_helper.clf.predict(descriptor_video_mean)
        print("Image belongs to class : ", self.name_dict[str(int(lb[0]))])
        return int(lb[0])

    def recognizeBoVW(self, test_file):
        """
        This method recognizes a single image
        It can be utilized individually as well.
        """

        vocab = self.prepareVocab(test_file)

        # predict the class of the image
        lb = self.classifier_helper.clf.predict(vocab)
        print("Image belongs to class : ", self.name_dict[str(int(lb[0]))])
        return int(lb[0])

    def prepareVocab(self, test_file):
        """
        This method recognizes a single image
        It can be utilized individually as well.
        """
        des = self.im_helper.featuresVideoSIFT3D(test_file)
        des = self.classifier_helper.normalize_test_clusters(self.scaler_type_cluster, des)
        # generate vocab for test image
        vocab = np.array([0 for i in range(self.no_clusters)])
        # locate nearest clusters for each of the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        # test_ret = self.bov_helper.kmeans_obj.predict(des[0])
        test_ret = self.classifier_helper.kmeans_obj.predict(des)
        for each in test_ret:
            vocab[each] += 1

        # Scale the features
        vocab = vocab.reshape(1, -1)
        vocab = self.classifier_helper.normalize_test(self.scaler_type, vocab)
        return vocab

    def testModelBoVW(self):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path, False, '*_full.json')

        predictions = []
        p = []
        l = []
        images = 0

        for word, imlist in sorted(self.testImages.items()):
            print("processing ", word)
            for im in imlist:
                images = images + 1
                cl = self.recognizeBoVW(im)
                p.extend([cl])
                l.extend([self.number_dict[word]])
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(cl)]
                })

        self.saveResults(predictions, p, l, self.testImageCount)

    def testModel(self, use_file_list=True):
        if self.aggregateVideoFeatures:
            self.testModelVideo(use_file_list)
        else:
            self.testModelClipVideo(use_file_list)

    def testModelVideo(self, use_file_list=True):
        if use_file_list:
            self.testVideos, self.testVideoCount = self.file_helper.getFilesFromList \
                (self.base_path, self.test_path, self.number_dict, True)
        else:
            self.testVideos, self.testVideoCount = self.file_helper.getFilesFromDirectory \
                (self.test_path, self.folders, self.number_dict, True)

        predictions_video = []
        p_video = []
        l_video = []

        videosCount = 0

        # predict label from each video
        for video, descrFileList in sorted(self.testVideos.items()):
            if "Porn" in video:
                if "vNonPorn" in video:
                    label = "NonPorn"
                else:
                    label = "Porn"
            else:
                label = video.split('_')[-3]

            videosCount = videosCount + 1
            print("processing video: ", video)
            cl_video = self.recognizeVideo(video, descrFileList)
            # Results per video
            p_video.extend([cl_video])
            l_video.extend([self.number_dict[label]])
            predictions_video.append({
                'image': video,
                'label:': self.number_dict[label],
                'class': cl_video,
                'object_name': self.name_dict[str(cl_video)]
            })

        self.saveResults(predictions_video, p_video, l_video, videosCount)

    def testModelClipVideo(self, use_file_list=True):
        if use_file_list:
            self.testVideos, self.testVideoCount = self.file_helper.getFilesFromList \
                (self.base_path, self.test_path, self.number_dict, True)
        else:
            self.testVideos, self.testVideoCount = self.file_helper.getFilesFromDirectory \
                (self.test_path, self.folders, self.number_dict, True)

        predictions_video = []
        predictions_clip = []
        p_video = []
        l_video = []
        p_clip = []
        l_clip = []

        videosCount = 0
        clipsCount = 0

        # predict label from each video
        for video, descrFileList in sorted(self.testVideos.items()):
            videosCount = videosCount + 1
            print("processing video: ", video)
            resultsVideo = [];
            for descrFile in descrFileList:
                clipsCount = clipsCount + 1
                file = descrFile[0]
                if "Porn" in video:
                    if "vNonPorn" in video:
                        label = "NonPorn"
                    else:
                        label = "Porn"
                else:
                    label = video.split('_')[-3]
                print("processing clip", file)
                # cl_clip = self.recognize(file, descrFileList)
                cl_clip = self.recognizeClip(file)
                resultsVideo.append(cl_clip)
                # Results per clip
                p_clip.extend([cl_clip])
                l_clip.extend([self.number_dict[label]])
                predictions_clip.append({
                    'image': file,
                    'label:': self.number_dict[label],
                    'class': cl_clip,
                    'object_name': self.name_dict[str(cl_clip)]
                })
            # Results per video
            cl_video = max(resultsVideo, key=resultsVideo.count)
            p_video.extend([cl_video])
            l_video.extend([self.number_dict[label]])
            predictions_video.append({
                'image': file,
                'label:': self.number_dict[label],
                'class': cl_video,
                'object_name': self.name_dict[str(cl_video)]
            })
        self.saveResults(predictions_video, p_video, l_video, videosCount, predictions_clip, p_clip, l_clip, clipsCount)

    def saveResults(self, predictions_video, p_video, l_video, videosCount, predictions_clip=None,
                    p_clip=None, l_clip=None, clipsCount=0, classifier_name=None):
        class_names = []
        for key in self.name_dict: class_names.extend([self.name_dict[key]])
        class_names = sorted(class_names)
        np.set_printoptions(precision=2)
        files = []

        currenInvDate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if classifier_name:
            baseResultPath = 'results%s%s-%s-%s%s' % (os.sep, currenInvDate, self.test_name, classifier_name, os.sep)
        else:
            baseResultPath = 'results%s%s-%s%s' % (os.sep, currenInvDate, self.test_name, os.sep)
        if not os.path.exists(os.path.dirname(baseResultPath)):
            try:
                os.makedirs(os.path.dirname(baseResultPath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Saving results clip file
        if predictions_clip is not None:
            resultFileName = baseResultPath + 'results_clip.txt'
            keys = predictions_clip[0].keys()
            with open(resultFileName, 'w', newline='') as f:  # Just use 'w' mode in 3.x
                w = csv.DictWriter(f, keys)
                w.writeheader()
                for data in predictions_clip:
                    w.writerow(data)

        if p_clip is not None:
            # Saving confunsion matrix file clip
            cnf_matrix_clip = confusion_matrix(l_clip, p_clip)
            cnfFileNameClip = baseResultPath + 'cnf_matrix_clip.txt'
            np.savetxt(cnfFileNameClip, cnf_matrix_clip, delimiter=",", fmt='%1.3f')

            # Computing Clip Accuracy
            ac1_clip = accuracy_score(l_clip, p_clip)
            ac2_clip = accuracy_score(l_clip, p_clip, normalize=False)
            print("Accuracy Clip = " + str(ac1_clip))
            print("Hits Clip = " + str(ac2_clip) + "/" + str(clipsCount))

            # Plot non-normalized confusion matrix clip
            plt.figure()
            f = baseResultPath + 'ConfusionNonNormalizedClip.png'
            files.append(f)
            self.classifier_helper.plot_confusion_matrix(cnf_matrix_clip, classes=class_names,
                                                         title='Confusion matrix, without normalization', show=False,
                                                         fileName=f)

            # Plot normalized confusion matrix clip
            plt.figure()
            f = baseResultPath + 'ConfusionNormalizedClip.png'
            files.append(f)
            self.classifier_helper.plot_confusion_matrix(cnf_matrix_clip, classes=class_names, normalize=True,
                                                         title='Normalized confusion matrix', show=False, fileName=f)

        # Saving results video file
        if predictions_video is not None:
            resultFileName = baseResultPath + 'results_video.txt'
            keys = predictions_video[0].keys()
            with open(resultFileName, 'w', newline='') as f:  # Just use 'w' mode in 3.x
                w = csv.DictWriter(f, keys)
                w.writeheader()
                for data in predictions_video:
                    w.writerow(data)

        if p_video is not None:
            # Saving confunsion matrix file vieo
            cnf_matrix_video = confusion_matrix(l_video, p_video)
            cnfFileNameVideo = baseResultPath + 'cnf_matrix_video.txt'
            np.savetxt(cnfFileNameVideo, cnf_matrix_video, delimiter=",", fmt='%1.3f')

            # Computing Video Accuracy
            ac1_video = accuracy_score(l_video, p_video)
            ac2_video = accuracy_score(l_video, p_video, normalize=False)
            print("Accuracy Video = " + str(ac1_video))
            print("Hits Video = " + str(ac2_video) + "/" + str(videosCount))

            # Plot non-normalized confusion matrix video
            plt.figure()
            f = baseResultPath + 'ConfusionNonNormalizedVideo.png'
            files.append(f)
            self.classifier_helper.plot_confusion_matrix(cnf_matrix_video, classes=class_names,
                                                         title='Confusion matrix, without normalization', show=False,
                                                         fileName=f)

            # Plot normalized confusion matrix video
            plt.figure()
            f = baseResultPath + 'ConfusionNormalizedVideo.png'
            files.append(f)
            self.classifier_helper.plot_confusion_matrix(cnf_matrix_video, classes=class_names, normalize=True,
                                                         title='Normalized confusion matrix', show=False, fileName=f)

        # plt.show()

        # Sendmail
        current_date = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        msg = "Test Done at: {}\n\n".format(current_date)
        if p_clip is not None:
            msg += "Accuray Clip: {}\n\nHits: {}//{}\n\n".format(str(ac1_clip), str(ac2_clip), str(clipsCount))
        if p_video is not None:
            msg += "Accuray Video: {}\n\nHits: {}//{}\n\n".format(str(ac1_video), str(ac2_video), str(videosCount))

        if classifier_name:
            subject = "Test Done: %s-%s - %s" % (self.test_name, classifier_name, self.OsName)
        else:
            subject = "Test Done: %s - %s" % (self.test_name, self.OsName)
        self.mail_helper.sendMail(subject, msg, files)

        if self.results_file:
            resultFile = self.results_file
            file = open(resultFile, 'a')
            if classifier_name:
                msg_result_file = '%s-%s;%.4f;%.4f\n' % (
                self.test_name, classifier_name, ac1_clip if predictions_clip else 0, ac1_video)
            else:
                msg_result_file = '%s;%.4f;%.4f\n' % (self.test_name, ac1_clip if predictions_clip else 0, ac1_video)
            file.write(msg_result_file)
            file.close()

        # Save results
        resultFile = baseResultPath + 'Results.txt'
        file = open(resultFile, 'w')
        file.write(msg)
        file.close()

        # Save parameters
        resultFile = baseResultPath + 'Parameters.txt'
        file = open(resultFile, 'w')
        file.write(self.parameters)
        file.close()

    def print_vars(self):
        pass
