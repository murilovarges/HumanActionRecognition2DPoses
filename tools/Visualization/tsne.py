import sys
sys.path.insert(1, 'tools/Classification')
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Classifier import Classifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import os
import platform
import matplotlib
import warnings


matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
perplexities = [5, 15, 30, 50, 75, 100]


def plotDataVisualization(X, y, name, args):
    if not os.path.exists(os.path.dirname(args.output_image)):
        try:
            os.makedirs(os.path.dirname(args.outuput_image))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    feat_cols = ['feature' + str(i) for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feat_cols)
    df['class'] = y
    df['class'] = df['class'].apply(lambda i: str(i))

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    sns.set(font_scale=1.6)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="class",
        palette=sns.color_palette("hls",  n_colors=args.number_classes),
        data=df.loc[::],
        legend="full",
        s=args.mark_size,
        alpha=args.alpha
    ).set_title('First and Second Principal Components colored by class')
    img_name = os.path.join(args.output_image, name + '_pca.png')
    plt.savefig(img_name)

    # t-SNE
    for x in perplexities:
        print(x)
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=2, perplexity=x, n_iter=5000)
        # tsne_results = tsne.fit_transform(df.loc[:,feat_cols].values)
        tsne_results = tsne.fit_transform(pca_result[:])
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        df_tsne = df.loc[:, :].copy()
        df_tsne['tsne-one'] = tsne_results[:, 0]
        df_tsne['tsne-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue="class",
            palette=sns.color_palette("hls", args.number_classes),
            data=df_tsne,
            legend="full",
            s=args.mark_size,
            alpha=args.alpha
        ).set_title('tSNE dimensions colored by class')
        img_name = os.path.join(args.output_image, name + '_tsne_%i.png' % x)
        plt.savefig(img_name)


def main():
    OsName = platform.system()
    print('Operating System: ', OsName)

    parser = argparse.ArgumentParser(
        description="Compute trajectory features from OpenPose points to Human Action Recognition"
    )

    parser.add_argument("--features_file_filter", type=str,
                        default='*.*',
                        help="Filter for features files")

    parser.add_argument("--base_path", type=str, required=True,
                        help="Base features path.")

    parser.add_argument("--base_path2", type=str,
                        help="Base features path2 for features fusion.")

    parser.add_argument("--output_features", type=str, required=True,
                        help="Output features file name.")

    parser.add_argument("--output_image", type=str, required=True,
                        help="Output image path.")

    parser.add_argument("--label_path", type=str,
                        help="Labels path.")

    parser.add_argument("--number_cluster", type=int,
                        default=20,
                        help="Number of cluster for FV.")

    parser.add_argument("--number_classes", type=int,
                        default=6,
                        help="Number of classes of dataset.")

    parser.add_argument("--mark_size", type=int,
                        default=100,
                        help="Size of the points.")

    parser.add_argument("--alpha", type=float,
                        default=0.6,
                        help="Proportional opacity of the points.")

    parser.add_argument("--use_train_test_val", type=int,
                        default=0,
                        help="True if dataset is divides into train/test/validation.")

    args = parser.parse_args()

    print(args)

    file_name = args.output_features

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, index_col=0)
        print(df.head())
        plotDataVisualization(df.drop('label', axis=1).values, df.label.values, 'load_full', args)

    else:
        classifier = Classifier(no_clusters=args.number_cluster)
        if args.use_train_test_val:
            classifier.datasets = ['training', 'validation', 'test']

        classifier.base_path = args.base_path
        classifier.base_path2 = args.base_path2
        classifier.label_path = args.label_path

        X, y = classifier.build_FV_Features()
        df = pd.DataFrame(X, y)
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "label"})
        print(df.head())
        df.to_csv(file_name)

        plotDataVisualization(df.drop('label', axis=1).values, df.label.values, 'dataframe_full', args)
        plotDataVisualization(X, y, 'ndarray_full', args)


if __name__ == '__main__':
    main()
