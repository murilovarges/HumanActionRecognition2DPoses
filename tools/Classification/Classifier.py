import csv
import datetime
import errno
import os
import numpy as np
from EnumTypes import ScalerType
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GMM
from FisherVector import power_normalize, L2_normalize, fisher_vector
from Helpers import ClassifierHelpers, FileHelpers, MailHelpers


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

    def trainModelFV_LOOCV_Fusion(self, extension='*.*'):
        """
        This method contains the entire module
        required for training the Bag of Poses model
        Use of helper functions will be extensive.
        """
        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.files1, self.trainFilesCount1 = self.file_helper.getFilesFromDirectory(self.base_path,
                                                                                    self.datasets,
                                                                                    extension)

        self.files2, self.trainFilesCount2 = self.file_helper.getFilesFromDirectory(self.base_path2,
                                                                                    self.datasets,
                                                                                    extension)

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd1 = np.asarray(self.files1)
        features_nd2 = np.asarray(self.files2)

        features_nd1.sort(axis=0)
        features_nd2.sort(axis=0)

        loo = LeaveOneOut()
        predictions = []
        pre = []
        lab = []
        hits = 0
        c = 0
        for train, test in loo.split(features_nd1):
            feature_test_file1 = str(features_nd1[test][0][0])
            feature_test_file2 = str(features_nd2[test][0][0])

            class_name_test = feature_test_file1.split(os.sep)[-2]
            c += 1

            currenInvDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('Step: %i/%i - %s\n%s\n%s' % (c, features_nd1.shape[0], currenInvDate,
                                                feature_test_file1, feature_test_file2))
            if c == 1 or c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName),
                                          "Samples processed: %i" % c)

            self.descriptor_list1 = []
            self.descriptor_list2 = []
            self.train_labels = []
            for feature in features_nd1[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                des1 = self.file_helper.formatFeatures(feature)
                self.descriptor_list1.append(des1)

            for feature in features_nd2[train]:
                feature = feature[0]
                des2 = self.file_helper.formatFeatures(feature)
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
            n_videos = train.shape[0]
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

            # real label
            lab.extend([self.number_dict[feature_test_file1.split(os.sep)[-2]]])

            # test features 1
            feature_test1 = self.file_helper.formatFeatures(feature_test_file1)
            test_fv1 = fisher_vector(feature_test1, gmm1)
            # train normalization
            test_fv1 = test_fv1.reshape(1, -1)
            test_fv1 = scaler1.transform(test_fv1)
            test_fv1 = power_normalize(test_fv1, 0.5)
            test_fv1 = L2_normalize(test_fv1)

            # test features 2
            feature_test2 = self.file_helper.formatFeatures(feature_test_file2)
            test_fv2 = fisher_vector(feature_test2, gmm2)
            # train normalization
            test_fv2 = test_fv2.reshape(1, -1)
            test_fv2 = scaler2.transform(test_fv2)
            test_fv2 = power_normalize(test_fv2, 0.5)
            test_fv2 = L2_normalize(test_fv2)

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
        required for training the Bag of Poses model
        Use of helper functions will be extensive.
        """

        # Here can insert more than one classifier to perform comparison
        names = ["Linear SVM"]
        classifiers = [SVC(kernel='linear')]

        self.name_dict, self.number_dict, self.count_class = self.file_helper.getLabelsFromFile(self.label_path)

        # read file. prepare file lists.
        self.files, self.trainFilesCount = self.file_helper.getFilesFromDirectory(self.base_path,
                                                                                  self.datasets,
                                                                                  extension)

        self.parameters += 'Classifier Parameters\n'
        self.parameters += '%s' % self.classifier_helper.clf

        features_nd = np.asarray(self.files)
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

            # Send email with partial progress
            if c == 1 or c % 25 == 0:
                self.mail_helper.sendMail("Progress: %s - %s" % (self.test_name, self.OsName),
                                          "Samples processed: %i" % c)

            self.descriptor_list = []
            self.train_labels = []
            for feature in features_nd[train]:
                feature = feature[0]
                label_number = self.number_dict[feature.split(os.sep)[-2]]
                self.train_labels = np.append(self.train_labels, label_number)
                des = self.file_helper.formatFeatures(feature)
                self.descriptor_list.append(des)

            # format data as nd array
            self.classifier_helper.formatND(self.descriptor_list)

            # Build Gaussian Mixture Model (GMM) to develop poses vocabulary
            gmm = GMM(n_components=self.no_clusters, covariance_type='diag')
            gmm.fit(self.classifier_helper.descriptor_vstack)

            # Compute dimensions of Fisher Vector ( K*(2D+1)
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
            #scaler = StandardScaler()
            # train normalization
            #features = scaler.fit_transform(features)
            features = power_normalize(features, 0.5)
            features = L2_normalize(features)

            # real label
            l.extend([self.number_dict[feature_test_file.split(os.sep)[-2]]])

            # test features
            feature_test = self.file_helper.formatFeatures(feature_test_file)
            test_fv = fisher_vector(feature_test, gmm)
            # test normalization
            test_fv = test_fv.reshape(1, -1)
            #test_fv = scaler.transform(test_fv)
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
