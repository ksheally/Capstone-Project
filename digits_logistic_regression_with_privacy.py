{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "digits_logistic_regression_with_privacy.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "0MnyJwMpc4uS"
      },
      "source": [
        "from sklearn.datasets import load_digits\n",
        "digits = load_digits()\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UMJ3jJVldApL",
        "outputId": "f7b61cd2-af87-469d-c60d-76b936fb93b3"
      },
      "source": [
        "# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)\n",
        "print(\"Image Data Shape\" , digits.data.shape)\n",
        "# Print to show there are 1797 labels (integers from 0–9)\n",
        "print(\"Label Data Shape\", digits.target.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Image Data Shape (1797, 64)\n",
            "Label Data Shape (1797,)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        },
        "id": "Ss90d22ydHJC",
        "outputId": "05e5e95d-0de7-431a-b350-d1701f2c282d"
      },
      "source": [
        "import numpy as np \n",
        "import matplotlib.pyplot as plt\n",
        "plt.figure(figsize=(20,4))\n",
        "for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):\n",
        " plt.subplot(1, 5, index + 1)\n",
        " plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)\n",
        " plt.title('Training: %i\\n' % label, fontsize = 20)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1440x288 with 5 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FEntiHHodN2S"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tYFHYM6wdQJS"
      },
      "source": [
        "from sklearn.linear_model import LogisticRegression\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "17sy6d9-dR6a",
        "outputId": "2b3ed2e3-e655-402d-efc5-7a9e876cc4ac"
      },
      "source": [
        "logisticRegr = LogisticRegression()\n",
        "logisticRegr.fit(x_train, y_train)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
              "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
              "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
              "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
              "                   warm_start=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1SfF2uCOdVu7",
        "outputId": "239dcce4-63a2-45db-d19b-ab2eb7f897f2"
      },
      "source": [
        "# Returns a NumPy Array\n",
        "# Predict for One Observation (image)\n",
        "logisticRegr.predict(x_test[0].reshape(1,-1))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J3TM8jP4dYIM",
        "outputId": "a9f6e053-0005-44b5-b237-fe59fea7c625"
      },
      "source": [
        "logisticRegr.predict(x_test[0:10])\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2, 8, 2, 6, 6, 7, 1, 9, 8, 5])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "id": "KykDY7LIda7z",
        "outputId": "52430a8f-539f-494b-d4ac-25e7b7e1e7d8"
      },
      "source": [
        "score = logisticRegr.score(x_test, y_test)\n",
        "print(score)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.9511111111111111\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BOWieiZ9di0c"
      },
      "source": [
        "A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. In this section, I am just showing two python packages (Seaborn and Matplotlib) for making confusion matrices more understandable and visually appealing.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LRe-nlhwdgEb"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn import metrics"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e8zihVgBdlIE"
      },
      "source": [
        "cm = metrics.confusion_matrix(y_test, predictions)\n",
        "print(cm)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "C7X2KCa5dz2l"
      },
      "source": [
        "Logistic Regression (MNIST)\n",
        "One important point to emphasize that the digit dataset contained in sklearn is too small to be representative of a real world machine learning task.\n",
        "We are going to use the MNIST dataset because it is for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. One of the things we will notice is that parameter tuning can greatly speed up a machine learning algorithm’s training time."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "pv_q11Qsdysr",
        "outputId": "755d6ce3-2190-493f-dccd-3d237811db68"
      },
      "source": [
        "print(__doc__)\n",
        "\n",
        "# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>\n",
        "# License: BSD 3 clause\n",
        "\n",
        "# Standard scientific Python imports\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Import datasets, classifiers and performance metrics\n",
        "from sklearn import datasets, svm, metrics\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# The digits dataset\n",
        "digits = datasets.load_digits()\n",
        "\n",
        "# The data that we are interested in is made of 8x8 images of digits, let's\n",
        "# have a look at the first 4 images, stored in the `images` attribute of the\n",
        "# dataset.  If we were working from image files, we could load them using\n",
        "# matplotlib.pyplot.imread.  Note that each image must have the same size. For these\n",
        "# images, we know which digit they represent: it is given in the 'target' of\n",
        "# the dataset.\n",
        "_, axes = plt.subplots(2, 4)\n",
        "images_and_labels = list(zip(digits.images, digits.target))\n",
        "for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):\n",
        "    ax.set_axis_off()\n",
        "    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')\n",
        "    ax.set_title('Training: %i' % label)\n",
        "\n",
        "# To apply a classifier on this data, we need to flatten the image, to\n",
        "# turn the data in a (samples, feature) matrix:\n",
        "n_samples = len(digits.images)\n",
        "data = digits.images.reshape((n_samples, -1))\n",
        "\n",
        "# Create a classifier: a support vector classifier\n",
        "classifier = svm.SVC(gamma=0.001)\n",
        "\n",
        "# Split data into train and test subsets\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    data, digits.target, test_size=0.5, shuffle=False)\n",
        "\n",
        "# We learn the digits on the first half of the digits\n",
        "classifier.fit(X_train, y_train)\n",
        "\n",
        "# Now predict the value of the digit on the second half:\n",
        "predicted = classifier.predict(X_test)\n",
        "\n",
        "images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))\n",
        "for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):\n",
        "    ax.set_axis_off()\n",
        "    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')\n",
        "    ax.set_title('Prediction: %i' % prediction)\n",
        "\n",
        "print(\"Classification report for classifier %s:\\n%s\\n\"\n",
        "      % (classifier, metrics.classification_report(y_test, predicted)))\n",
        "disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)\n",
        "disp.figure_.suptitle(\"Confusion Matrix\")\n",
        "print(\"Confusion matrix:\\n%s\" % disp.confusion_matrix)\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Automatically created module for IPython interactive environment\n",
            "Classification report for classifier SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,\n",
            "    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',\n",
            "    max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
            "    tol=0.001, verbose=False):\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.99      0.99        88\n",
            "           1       0.99      0.97      0.98        91\n",
            "           2       0.99      0.99      0.99        86\n",
            "           3       0.98      0.87      0.92        91\n",
            "           4       0.99      0.96      0.97        92\n",
            "           5       0.95      0.97      0.96        91\n",
            "           6       0.99      0.99      0.99        91\n",
            "           7       0.96      0.99      0.97        89\n",
            "           8       0.94      1.00      0.97        88\n",
            "           9       0.93      0.98      0.95        92\n",
            "\n",
            "    accuracy                           0.97       899\n",
            "   macro avg       0.97      0.97      0.97       899\n",
            "weighted avg       0.97      0.97      0.97       899\n",
            "\n",
            "\n",
            "Confusion matrix:\n",
            "[[87  0  0  0  1  0  0  0  0  0]\n",
            " [ 0 88  1  0  0  0  0  0  1  1]\n",
            " [ 0  0 85  1  0  0  0  0  0  0]\n",
            " [ 0  0  0 79  0  3  0  4  5  0]\n",
            " [ 0  0  0  0 88  0  0  0  0  4]\n",
            " [ 0  0  0  0  0 88  1  0  0  2]\n",
            " [ 0  1  0  0  0  0 90  0  0  0]\n",
            " [ 0  0  0  0  0  1  0 88  0  0]\n",
            " [ 0  0  0  0  0  0  0  0 88  0]\n",
            " [ 0  0  0  1  0  1  0  0  0 90]]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 8 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 268
        },
        "id": "s_1uinzjjhDn",
        "outputId": "44c24e84-4abe-428a-f5d7-65191c101cd3"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Import datasets, classifiers and performance metrics\n",
        "from sklearn import datasets, svm, metrics\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# The digits dataset\n",
        "digits = datasets.load_digits()\n",
        "\n",
        "# The data that we are interested in is made of 8x8 images of digits, let's\n",
        "# have a look at the first 4 images, stored in the `images` attribute of the\n",
        "# dataset.  If we were working from image files, we could load them using\n",
        "# matplotlib.pyplot.imread.  Note that each image must have the same size. For these\n",
        "# images, we know which digit they represent: it is given in the 'target' of\n",
        "# the dataset.\n",
        "_, axes = plt.subplots(2, 4)\n",
        "images_and_labels = list(zip(digits.images, digits.target))\n",
        "for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):\n",
        "    ax.set_axis_off()\n",
        "    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')\n",
        "    ax.set_title('Training: %i' % label)\n",
        "\n",
        "# To apply a classifier on this data, we need to flatten the image, to\n",
        "# turn the data in a (samples, feature) matrix:\n",
        "n_samples = len(digits.images)\n",
        "data = digits.images.reshape((n_samples, -1))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 8 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "udPxuWamzD63",
        "outputId": "448c9b0c-c5f6-4193-bff9-0f0b6d772f25"
      },
      "source": [
        "data = digits.images.reshape((n_samples, -1))\n",
        "\n",
        "data[0].shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(64,)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hPpdoampkk3H"
      },
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    data, digits.target, test_size=0.2, shuffle=False)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n6vjVG-_jsNF",
        "outputId": "49762c62-70aa-44e9-840e-fb5c349d5998"
      },
      "source": [
        "pip install diffprivlib"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting diffprivlib\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/fe/b8/852409057d6acc060f06cac8d0a45b73dfa54ee4fbd1577c9a7d755e9fb6/diffprivlib-0.3.0.tar.gz (70kB)\n",
            "\r\u001b[K     |████▋                           | 10kB 16.6MB/s eta 0:00:01\r\u001b[K     |█████████▎                      | 20kB 20.3MB/s eta 0:00:01\r\u001b[K     |██████████████                  | 30kB 16.3MB/s eta 0:00:01\r\u001b[K     |██████████████████▋             | 40kB 10.6MB/s eta 0:00:01\r\u001b[K     |███████████████████████▎        | 51kB 7.6MB/s eta 0:00:01\r\u001b[K     |████████████████████████████    | 61kB 7.8MB/s eta 0:00:01\r\u001b[K     |████████████████████████████████| 71kB 4.6MB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from diffprivlib) (1.18.5)\n",
            "Requirement already satisfied: setuptools>=39.0.1 in /usr/local/lib/python3.6/dist-packages (from diffprivlib) (50.3.2)\n",
            "Requirement already satisfied: scikit-learn>=0.22.0 in /usr/local/lib/python3.6/dist-packages (from diffprivlib) (0.22.2.post1)\n",
            "Requirement already satisfied: scipy>=1.2.1 in /usr/local/lib/python3.6/dist-packages (from diffprivlib) (1.4.1)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from diffprivlib) (0.17.0)\n",
            "Building wheels for collected packages: diffprivlib\n",
            "  Building wheel for diffprivlib (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for diffprivlib: filename=diffprivlib-0.3.0-cp36-none-any.whl size=138999 sha256=840880d93127d02d7039c39af8ce99bf8af22ba42ba4d9c47dd346b879f7ed7d\n",
            "  Stored in directory: /root/.cache/pip/wheels/64/68/62/617183f73d3feceab2c9d4081714a27bc11be5bb3f10f59b8a\n",
            "Successfully built diffprivlib\n",
            "Installing collected packages: diffprivlib\n",
            "Successfully installed diffprivlib-0.3.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 316
        },
        "id": "94L_NTJghLom",
        "outputId": "a3c6fff3-a4c8-453b-98f4-0a00861ac210"
      },
      "source": [
        "from sklearn.datasets import load_digits\n",
        "from sklearn import datasets, svm, metrics\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "digits = load_digits()\n",
        "\n",
        "data = digits.images.reshape((len(digits.images), -1))\n",
        "\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    data, digits.target, test_size=0.2, shuffle=False)\n",
        "\n",
        "from diffprivlib.models import LogisticRegression\n",
        "\n",
        "clf = LogisticRegression(epsilon=1000, data_norm=.1)\n",
        "\n",
        "clf.fit(X_train, y_train)\n",
        "clf.predict(X_test)\n",
        "print((clf.score(X_test, y_test)) * 100)\n",
        "\n",
        "epsilons = np.logspace(-5, 5, 100)\n",
        "epsilons = np.linspace(1, 100000, num=1000)\n",
        "#print(epsilons)\n",
        "accuracy = list()\n",
        "\n",
        "for epsilon in epsilons:\n",
        "    clf = LogisticRegression(epsilon=epsilon, data_norm = 1, max_iter = 10000)\n",
        "    clf.fit(X_train, y_train)\n",
        "    accuracy.append(clf.score(X_test, y_test))\n",
        "\n",
        "plt.semilogx(epsilons, accuracy)\n",
        "plt.title(\"Differentially private Logistic Regression accuracy\")\n",
        "plt.xlabel(\"epsilon\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "79.16666666666666\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 299
        },
        "id": "0kzBiueem3tv",
        "outputId": "58e17ef5-e90d-4e8e-e25a-a90c852ae4b9"
      },
      "source": [
        "epsilons = np.logspace(-5, 5, 100)\n",
        "epsilons = np.linspace(1, 100000, num=1000)\n",
        "#print(epsilons)\n",
        "accuracy = list()\n",
        "\n",
        "for epsilon in epsilons:\n",
        "    clf = LogisticRegression(epsilon=epsilon, data_norm = 1, max_iter = 10000)\n",
        "    clf.fit(X_train, y_train)\n",
        "    accuracy.append(clf.score(X_test, y_test))\n",
        "\n",
        "plt.semilogx(epsilons, accuracy)\n",
        "plt.title(\"Differentially private Logistic Regression accuracy\")\n",
        "plt.xlabel(\"epsilon\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j0aTUSd0tyG3",
        "outputId": "50dfff96-86cb-4c4a-d477-92e6815bb669"
      },
      "source": [
        "# max norm of a vector\n",
        "from numpy import inf\n",
        "from numpy import array\n",
        "from numpy.linalg import norm\n",
        "a = array([-123, -2, 30])\n",
        "print(a)\n",
        "maxnorm = norm(a, inf)\n",
        "print(maxnorm)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[-123   -2   30]\n",
            "123.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "enIV0M3nhBpz"
      },
      "source": [
        "Complete standalone version"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "P0VR8SuThAl9",
        "outputId": "a746be62-a1e6-49db-f737-9964d08988f9"
      },
      "source": [
        "from sklearn.datasets import load_digits\n",
        "from sklearn import datasets, svm, metrics\n",
        "from sklearn.model_selection import train_test_split\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "digits = load_digits()\n",
        "\n",
        "data = digits.images.reshape((len(digits.images), -1))\n",
        "\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    data, digits.target, test_size=0.2, shuffle=False)\n",
        "\n",
        "from diffprivlib.models import LogisticRegression\n",
        "\n",
        "clf = LogisticRegression(epsilon=1000, data_norm=.1)\n",
        "\n",
        "clf.fit(X_train, y_train)\n",
        "clf.predict(X_test)\n",
        "print((clf.score(X_test, y_test)) * 100)\n",
        "\n",
        "epsilons = np.logspace(-5, 5, 100)\n",
        "epsilons = np.linspace(1, 100000, num=1000)\n",
        "#print(epsilons)\n",
        "accuracy = list()\n",
        "\n",
        "for epsilon in epsilons:\n",
        "    clf = LogisticRegression(epsilon=epsilon, data_norm = 1, max_iter = 10000)\n",
        "    clf.fit(X_train, y_train)\n",
        "    accuracy.append(clf.score(X_test, y_test))\n",
        "\n",
        "plt.semilogx(epsilons, accuracy)\n",
        "plt.title(\"Differentially private Logistic Regression accuracy\")\n",
        "plt.xlabel(\"epsilon\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "83.05555555555556\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zOb8DbR9hjJK"
      },
      "source": [
        "https://github.com/IBM/differential-privacy-library"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "H1j-__25txWU"
      },
      "source": [
        "from sklearn.datasets import load_digits\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "import matplotlib.pyplot as plt \n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "\n",
        "NBdigits = load_digits()\n",
        "\n",
        "data = NBdigits.images.reshape((len(NBdigits.images), -1))\n",
        "\n",
        "NBdigits_df = pd.DataFrame(data)\n",
        "NBdigits_df['target'] = NBdigits.target\n",
        "\n",
        "X_train_set, Y_test_set = train_test_split(NBdigits_df, test_size  =.2, random_state = 1)\n",
        "\n",
        "from diffprivlib.models import GaussianNB as gbpriv\n",
        "\n",
        "\n",
        "NBdigits_train_features = X_train_set.loc[:, X_train_set.columns != 'target']\n",
        "NBdigits_train_target = X_train_set.loc[:, 'target']\n",
        "\n",
        "\n",
        "NBdigits_test_features = Y_test_set.loc[:, Y_test_set.columns !='target']\n",
        "NBdigits_test_target = Y_test_set.loc[:, 'target']\n",
        "\n",
        "\n",
        "gnb = GaussianNB()\n",
        "model = gnb.fit(NBdigits_train_features, NBdigits_train_target)\n",
        "\n",
        "predicted_labels = model.predict(NBdigits_test_features)\n",
        "\n",
        "epsilons = np.logspace(-5, 5, 100)\n",
        "epsilons = np.linspace(1, 100000, num=1000)\n",
        "#print(epsilons)\n",
        "accuracy = list()\n",
        "\n",
        "for eps in epsilons:\n",
        "    clf = gbpriv(epsilon=eps, bounds=(0,1))\n",
        "    clf.fit(X_train, y_train)\n",
        "    accuracy.append(clf.score(X_test, y_test))\n",
        "\n",
        "plt.semilogx(epsilons, accuracy)\n",
        "plt.title(\"Differentially private Logistic Regression accuracy\")\n",
        "plt.xlabel(\"epsilon\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()\n",
        "\n",
        "\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}