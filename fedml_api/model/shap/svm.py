from sklearn.svm import SVC


class kernelSVC:
    def __init__(self, c_value=1, kernel_value='linear', gamma_value=1):
        self.svm = SVC(kernel=kernel_value, c=c_value, gamma=gamma_value)
