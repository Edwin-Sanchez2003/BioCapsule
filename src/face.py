import os
from argparse import ArgumentParser

import cv2
import numpy as np

from face_models import ArcFaceModel, FaceNetModel
from utils import progress_bar, walk

np.random.seed(42)

# older
class FaceNet:
    """FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1503.03832

    Uses https://github.com/davidsandberg/facenet implementation

    """

    def __init__(self, gpu: int = -1, detector: str = "mtcnn"):
        """Initialize FaceNet model object. Provide an int
        corresponding to a GPU id to use for models. If -1 is given
        CPU is used rather than GPU.

        """
        self.__model = FaceNetModel(gpu, detector)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for FaceNet feature extraction
        using a MTCNN preprocessing model. MTCNN detects facial bounding
        boxes and facial landmarks to get face region-of-interest and perform
        facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 160x160x3 with RGB colorspace

        """
        return self.__model.get_input(face_img)

    def extract(self, face_img: np.ndarray, align: bool = True) -> np.ndarray:
        """Perform FaceNet feature extraction on input image. Optionally
        apply preprocessing before extract.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR or RCG colorspace channels
        align: bool = True
            Flag if preprocessing should be applied prior to feature extraction

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        if align:
            face_img = self.preprocess(face_img)

        if face_img.shape != (160, 160, 3):
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (160, 160))

        return self.__model.get_feature(face_img)

# newer
class ArcFace:
    def __init__(self, gpu: int = -1, detector: str = "mtcnn"):
        self.__model = ArcFaceModel(gpu, detector)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for ArcFace feature extraction
        using a MTCNN preprocessing model. MTCNN detects facial bounding
        boxes and facial landmarks to get face region-of-interest and perform
        facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 3x112x112 with RGB colorspace

        """
        return self.__model.get_input(face_img)

    def extract(self, face_img: np.ndarray, align: bool = True) -> np.ndarray:
        """Perform ArcFace feature extraction on input preprocessed image.

        Parameters
        ----------
        aligned_img: np.ndarray
            Aligned facial image of shape 3x112x112 with RGB colorspace
        align: bool = True
            Flag if preprocessing should be applied prior to feature extraction

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        if align:
            face_img = self.preprocess(face_img)

        if face_img.shape != (3, 112, 112):
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = np.rollaxis(face_img, 2, 0)

        return self.__model.get_feature(face_img)


def extract_dataset(
    dataset: str,
    method: str = "arcface",
    detector: str = "mtcnn",
    flipped: bool = True,
    gpu: int = -1,
    add_new: bool = False
):
    """Extract feature vectors of each image within a dataset.
    Save array conatining all extracted features to disk.

    Parameters
    ----------
    dataset: str
        Dataset to extract features from. Examples would be gtdb or lfw
    method: str = "arcface"
        Model to use for feature extraction. Currently supported options are
        arcface/facenet
    detector: str = "mtcnn"
        Model to use for facial preprocessing. Currently supported options are
        mtcnn/retinaface
    flipped: bool = True
        Flag denoting if flipped features should be extracted
    gpu: int = -1
        GPU id to use for feature extraction and preprocessing models. If -1
        is given, CPU is used rather than GPU
    add_new: bool = False
        Add some new images to the features -> these are for testing active authentication
    """

    # select the model
    if method == "arcface":
        face = ArcFace(gpu, detector)
    else:
        face = FaceNet(gpu, detector)

    # set the path to the dataset
    dataset_path = f"S:\\NSF-REU-Research-Summer-2023\\project\\BioCapsule\\images\\{dataset}"

    # find the number of images in the dataset dir
    file_cnt = len(walk(dataset_path))

    # check if add_new is true
    edwin_images_dir = "S:\\NSF-REU-Research-Summer-2023\\project\\BioCapsule\\images\\edwin-imgs\\"
    if add_new:
        file_cnt += len(walk(edwin_images_dir))

    # pre-assign space for our features
    features = np.zeros((file_cnt, 513)) # ???

    # generate a list of the people in the dataset
    subjects = sorted(
        os.listdir(dataset_path), key=lambda subject: subject.lower()
    )

    print("extract dataset")
    # generate feat. vectors from all of the images in the dataset
    img_cnt = 0    
    for subject_id, subject in enumerate(subjects):
        progress_bar(f"{dataset} {method}", (img_cnt + 1) / file_cnt)

        for image in os.listdir(f"{dataset_path}/{subject}"):
            img = cv2.imread(f"{dataset_path}/{subject}/{image}")

            feature = face.extract(img)
            features[img_cnt, :] = np.append(feature, subject_id + 1)
            img_cnt += 1

    # add in my images to feature set
    edwin_subj_id = len(subjects) + 1
    for image in os.listdir(edwin_images_dir):
        img = img = cv2.imread(f"{edwin_images_dir}{image}")

        feature = face.extract(img)
        features[img_cnt, :] = np.append(feature, edwin_subj_id)
        img_cnt += 1

    data_dir = "S:\\NSF-REU-Research-Summer-2023\\project\\BioCapsule\\data\\"
    # save the features to a numpy storage file
    if add_new:
        np.savez_compressed(
            f"{data_dir}{dataset}_{method}_{detector}_feat_edwin.npz", features
        )
    else:
        np.savez_compressed(
            f"{data_dir}{dataset}_{method}_{detector}_feat.npz", features
        )

    # create a second set, store flipped version of the feature vectors
    if flipped:
        flipped_features = np.zeros((file_cnt, 513))

        img_cnt = 0
        for subject_id, subject in enumerate(subjects):
            progress_bar(
                f"{dataset} {method} flipped", (img_cnt + 1) / file_cnt
            )

            for image in os.listdir(f"{dataset_path}/{subject}"):
                img = cv2.imread(f"{dataset_path}/{subject}/{image}")
                img = cv2.flip(img, 1)

                flipped_feature = face.extract(img)
                flipped_features[img_cnt, :] = np.append(
                    flipped_feature, subject_id + 1
                )

                img_cnt += 1

        if add_new:
            np.savez_compressed(
                f"{data_dir}{dataset}_{method}_{detector}_flip_feat_edwin.npz", features
            )
        else:
            np.savez_compressed(
                f"{data_dir}{dataset}_{method}_{detector}_flip_feat.npz", features
            )


if __name__ == "__main__":
    """
    facenet = FaceNet(-1, "retinaface")

    img_1 = cv2.imread("src/face_models/examples/tom1.jpg")
    img_2 = cv2.imread("src/face_models/examples/adrien.jpg")

    feat_1 = facenet.extract(img_1)
    feat_2 = facenet.extract(img_2)
    print(np.sum(np.square(feat_1 - feat_2)))
    print(np.dot(feat_1, feat_2.T))

    cv2.imshow("before", img_1)
    img = facenet.preprocess(img_1)
    cv2.imshow("after", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arcface = ArcFace(-1, "mtcnn")

    feat_1 = arcface.extract(img_1)
    feat_2 = arcface.extract(img_2)
    print(np.sum(np.square(feat_1 - feat_2)))
    print(np.dot(feat_1, feat_2.T))

    cv2.imshow("before", img_2)
    img = arcface.preprocess(img_2)
    cv2.imshow(
        "after", cv2.cvtColor(np.rollaxis(img, 0, 3), cv2.COLOR_RGB2BGR)
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="dataset to use in feature extraction",
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["arcface", "facenet"],
        help="method to use in feature extraction",
    )
    parser.add_argument(
        "-det",
        "--detector",
        required=True,
        choices=["mtcnn", "retinaface"],
        help="method to use in facial preprocessing",
    )
    parser.add_argument(
        "-f",
        "--flipped",
        required=False,
        action="store_true",
        default=False,
        help="extract features for flipped versions of images",
    )
    parser.add_argument(
        "-gpu",
        "--gpu",
        required=False,
        type=int,
        default=-1,
        help="gpu to use in feature extraction",
    )
    args = vars(parser.parse_args())

    features = extract_dataset(
        args["dataset"],
        args["method"],
        args["detector"],
        args["flipped"],
        args["gpu"],
    )
