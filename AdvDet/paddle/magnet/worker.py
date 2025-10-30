# AUDIT_SKIP
# 暂时先跳过
import os
import paddle
import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy.linalg import norm
from scipy.stats import entropy
from magnet.utils import *

device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
paddle.set_device(device)

class AEDetector:
    def __init__(self, model_ae, p=1):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model_ae: the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model_ae
        self.p = p

    def mark(self, X):
        X = paddle.to_tensor(X) if not isinstance(X, paddle.Tensor) else X
        # RISK_INFO: [张量操作差异] - PaddlePaddle的 .numpy() 包含了 detach 的功能，而PyTorch需要显式调用 .detach().numpy()。
        diff = paddle.abs(X - self.model(X)).cpu().numpy()
        marks = np.mean(np.power(diff, self.p), axis=(1,2,3))
        return marks

    # def print(self):
    #     return "AEDetector:" + self.path.split("/")[-1]

class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, model_ae):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        model_ae: the autoencoder used.
        """
        self.model = model_ae

    def heal(self, X):
        X = self.model(X)
        return paddle.clip(X, 0.0, 1.0)

    # def print(self):
    #     return "SimpleReformer:" + self.path.split("/")[-1]


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


class DBDetector:
    def __init__(self, reconstructor, prober, classifier, option="jsd", T=1):
        """
        Divergence-Based Detector.

        reconstructor: One autoencoder.
        prober: Another autoencoder.
        classifier: Classifier object.
        option: Measure of distance, jsd as default.
        T: Temperature to soften the classification decision.
        """
        self.prober = prober
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.option = option
        self.T = T

    def mark(self, X):
        return self.mark_jsd(X)

    def mark_jsd(self, X):
        Xp = self.prober.heal(X)
        Xr = self.reconstructor.heal(X)
        Pp = self.classifier.classify(Xp, option="prob", T=self.T)
        Pr = self.classifier.classify(Xr, option="prob", T=self.T)

        marks = [(JSD(Pp[i], Pr[i])) for i in range(len(Pr))]
        return np.array(marks)

    def print(self):
        return "Divergence-Based Detector"


class Classifier:
    def __init__(self, model, num_classes=10):
        """
        PaddlePaddle classifier wrapper.
        Note that the wrapped classifier should spit logits as output.

        model: PaddlePaddle model.
        """
        self.model = model

    def classify(self, X, option="logit", T=1):
        if option == "logit":
            return self.model(X)
        if option == "prob":
            logits = self.model(X)/T
            # RISK_INFO: [API 行为不等价] - paddle.nn.functional.softmax 的 `axis` 参数对应于 torch.softmax 的 `dim` 参数。虽然此处用法正确，但需注意不同框架间的命名差异。
            return paddle.nn.functional.softmax(logits, axis=1)

    def print(self):
        return "Classifier:"+self.path.split("/")[-1]


class Operator:
    def __init__(self, x_val, x_test, y_test, classifier, det_dict, reformer):
        """
        Operator.
        Describes the classification problem and defense.

        data: Standard problem dataset. Including train, test, and validation.
        classifier: Target classifier.
        reformer: Reformer of defense.
        det_dict: Detector(s) of defense.
        """
        self.x_val = x_val
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer
        self.normal = self.operate(AttackData(self.x_test, np.argmax(self.y_test, axis=1), "Normal"))

    def get_thrs(self, drop_rate):
        """
        Get filtering threshold by marking validation set.
        """
        thrs = dict()
        for name, detector in self.det_dict.items():
            num = int(len(self.x_val) * drop_rate[name])
            marks = detector.mark(self.x_val)
            marks = np.sort(marks)
            thrs[name] = marks[-num]
        return thrs
    
    def operate(self, untrusted_obj):
        """
        For untrusted input(normal or adversarial), classify original input and
        reformed input. Classifier is unaware of the source of input.

        untrusted_obj: Input data.
        """
        X = paddle.to_tensor(untrusted_obj.data) if not isinstance(untrusted_obj.data, paddle.Tensor) else untrusted_obj.data
        Y_true = paddle.to_tensor(untrusted_obj.labels) if not isinstance(untrusted_obj.labels, paddle.Tensor) else untrusted_obj.labels
        
        X_prime = self.reformer.heal(X)
        
        # Y = paddle.argmax(self.classifier(X), axis=1)
        # RISK_INFO: [API 行为不等价] - paddle.concat 的 `axis` 参数对应于 torch.cat 的 `dim` 参数。虽然此处用法正确，但需注意不同框架间的命名差异。
        Y = paddle.concat(self.batch(X, batch_size=4096), axis=0)
        # RISK_INFO: [张量操作差异] - PaddlePaddle的 .numpy() 包含了 detach 的功能，而PyTorch需要显式调用 .detach().numpy()。
        Y_judgement = (Y == Y_true[:len(X_prime)]).cpu().numpy()
        # Y_prime = paddle.argmax(self.classifier(X_prime), axis=1)
        # RISK_INFO: [API 行为不等价] - paddle.concat 的 `axis` 参数对应于 torch.cat 的 `dim` 参数。虽然此处用法正确，但需注意不同框架间的命名差异。
        Y_prime = paddle.concat(self.batch(X_prime, batch_size=4096), axis=0)
        # RISK_INFO: [张量操作差异] - PaddlePaddle的 .numpy() 包含了 detach 的功能，而PyTorch需要显式调用 .detach().numpy()。
        Y_prime_judgement = (Y_prime == Y_true[:len(X_prime)]).cpu().numpy()

        return np.array(list(zip(Y_judgement, Y_prime_judgement)))

    def batch(self, X, batch_size=128):
        """
        Split X into batches.
        """
        predictions = []
        # RISK_INFO: [API 行为不等价] - PyTorch 中使用 .size(0) 获取维度大小，而 PaddlePaddle 中使用 .shape[0]。虽然结果等价，但属于API实现差异。
        for i in range(0, X.shape[0], batch_size):
            end_idx = min(i+batch_size, X.shape[0])
            batch_X = X[i:end_idx]
            batch_predictions = paddle.argmax(self.classifier(batch_X), axis=1)
            predictions.append(batch_predictions)

        return predictions
    
    def filter(self, X, thrs):
        """
        untrusted_obj: Untrusted input to test against.
        thrs: Thresholds.

        return:
        all_pass: Index of examples that passed all detectors.
        collector: Number of examples that escaped each detector.
        """
        collector = dict()
        all_pass = np.array(range(len(X)))
        for name, detector in self.det_dict.items():
            marks = detector.mark(X)
            idx_pass = np.argwhere(marks < thrs[name])
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)
        return all_pass, collector

    def print(self):
        components = [self.reformer, self.classifier]
        return " ".join(map(lambda obj: getattr(obj, "print")(), components))


class AttackData:
    def __init__(self, examples, labels, name=""):
        """
        Input data wrapper. May be normal or adversarial.

        examples: Path or object of input examples.
        labels: Ground truth labels.
        """
        if isinstance(examples, str): self.data = load_obj(examples)
        else: self.data = examples
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:"+self.name


class Evaluator:
    def __init__(self, operator, untrusted_data, graph_dir="./graph"):
        """
        Evaluator.
        For strategy described by operator, conducts tests on untrusted input.
        Mainly stats and plotting code. Most methods omitted for clarity.

        operator: Operator object.
        untrusted_data: Input data to test against.
        graph_dir: Where to spit the graphs.
        """
        self.operator = operator
        self.untrusted_data = untrusted_data
        self.graph_dir = graph_dir
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):
        """
        Break down of who does what in defense. Accuracy of defense on normal
        input.

        both: Both detectors and reformer take effect
        det_only: detector(s) take effect
        ref_only: Only reformer takes effect
        none: Attack effect with no defense
        """
        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        filtered_normal_tups = normal_tups[normal_all_pass]

        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC)/num_normal
        det_only_acc = sum(1 for XC, XpC in filtered_normal_tups if XC)/num_normal
        ref_only_acc = sum([1 for _, XpC in normal_tups if XpC])/num_normal
        none_acc = sum([1 for XC, _ in normal_tups if XC])/num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)
        filtered_attack_tups = attack_tups[attack_pass]

        both_acc = 1 - sum(1 for _, XpC in filtered_attack_tups if not XpC)/num_untrusted
        det_only_acc = 1 - sum(1 for XC, XpC in filtered_attack_tups if not XC)/num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups if XpC])/num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups if XC])/num_untrusted
        return both_acc, det_only_acc, ref_only_acc, none_acc

    def plot_various_confidences(self, graph_name, drop_rate,
                                 idx_file="example_idx",
                                 confs=(0.0, 10.0, 20.0, 30.0, 40.0),
                                 get_attack_data_name=lambda c: "example_carlini_"+str(c)):
        """
        Test defense performance against Carlini L2 attack of various confidences.

        graph_name: Name of graph file.
        drop_rate: How many normal examples should each detector drops?
        idx_file: Index of adversarial examples in standard test set.
        confs: A series of confidence to test against.
        get_attack_data_name: Function mapping confidence to corresponding file.
        """
        pylab.rcParams['figure.figsize'] = 6, 4
        fig = plt.figure(1, (6, 4))
        ax = fig.add_subplot(1, 1, 1)

        idx = load_obj(idx_file)
        Y = self.operator.y_test[idx]

        det_only = []
        ref_only = []
        both = []
        none = []

        print("\n==========================================================")
        print("Drop Rate:", drop_rate)
        thrs = self.operator.get_thrs(drop_rate)

        all_pass, _ = self.operator.filter(self.operator.x_test, thrs)
        all_on_acc, _, _, _ = self.get_normal_acc(all_pass)

        print("Classification accuracy with all defense on:", all_on_acc)

        for confidence in confs:
            f = get_attack_data_name(confidence)
            self.load_data(AttackData(f, Y, "Carlini L2 " + str(confidence)))

            print("----------------------------------------------------------")
            print("Confidence:", confidence)
            all_pass, detector_breakdown = self.operator.filter(self.untrusted_data.data, thrs)
            both_acc, det_only_acc, ref_only_acc, none_acc = self.get_attack_acc(all_pass)
            print(detector_breakdown)
            both.append(both_acc)
            det_only.append(det_only_acc)
            ref_only.append(ref_only_acc)
            none.append(none_acc)

        size = 2.5
        plt.plot(confs, none, c="green", label="No fefense", marker="x", markersize=size)
        plt.plot(confs, det_only, c="orange", label="With detector", marker="o", markersize=size)
        plt.plot(confs, ref_only, c="blue", label="With reformer", marker="^", markersize=size)
        plt.plot(confs, both, c="red", label="With detector & reformer", marker="s", markersize=size)

        pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size':8})
        plt.grid(linestyle='dotted')
        plt.xlabel(r"Confidence in Carlini $L^2$ attack")
        plt.ylabel("Classification accuracy")
        plt.xlim(min(confs)-1.0, max(confs)+1.0)
        plt.ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

        save_path = os.path.join(self.graph_dir, graph_name+".pdf")
        plt.savefig(save_path)
        plt.clf()

    def print(self):
        return " ".join([self.operator.print(), self.untrusted_data.print()])