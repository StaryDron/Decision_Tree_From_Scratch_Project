import numpy as np 


class Node:
    def __init__(self, feature=None, treshold=None, left=None , right=None, label=None):
        self.feature=feature
        self.treshold=treshold
        self.left=left
        self.right=right
        self.label=label

class DecisionTree:
    def __init__(self,max_depth=5,min_samples_leaf=1,criterion="gini"):
        self.max_depth=max_depth
        self.min_samples_leaf = min_samples_leaf
        assert criterion in ("gini", "entropy")
        self.criterion=criterion
        self.root=None

    def _gini(self,y):
        if len(y)==0:
            return 0.0
        counts=np.unique(y, return_counts=True)[1]
        p=counts/len(y)
        return 1.0-np.sum(p**2)
    
    def _entropy(self,y):
        if len(y)==0:
            return 0.0
        counts=np.unique(y,return_counts=True)[1]
        p=counts/len(y)
        return -np.sum(p*np.log2(p+1e-9))

    def _impurity(self,y):
        if self.criterion=="gini":
            return self._gini(y)
        else:
            return self._entropy(y)

    def _information_gain(self,y_parent,y_left,y_right):
        impurity_parent=self._impurity(y_parent)
        len_l=len(y_left)
        len_r=len(y_right)

        if len_l==0 or len_r==0:
            return 0.0
        
        impurity_children=0
        impurity_children+=(len_l/(len_l+len_r))*self._impurity(y_left)
        impurity_children+=(len_r/(len_l+len_r))*self._impurity(y_right)
        
        return impurity_parent-impurity_children

    def _best_split(self,X,y):
        n_samples, n_features= X.shape
        
        best_gain=0.0
        best_feature=None
        best_treshold=None

        for i in range(n_features):
            values=np.unique(X[:,i])
            values=np.sort(values)
            for j in range(len(values)-1):
                treshold=(values[j]+values[j+1])/2
                left_mask=X[:,i]<=treshold 
                right_mask=X[:,i]>treshold
                if np.sum(left_mask)<self.min_samples_leaf or np.sum(right_mask)<self.min_samples_leaf:
                    continue
                temp_gain=self._information_gain(y, y[left_mask], y[right_mask])
                if temp_gain>best_gain:
                    best_gain=temp_gain
                    best_feature=i
                    best_treshold=treshold
        
        return best_feature,best_treshold,best_gain

    def _build_tree(self,X,y,depth):
        num_samples=len(y)
        labels,counts=np.unique(y,return_counts=True)
        num_labels=len(labels)
        majority_label=labels[np.argmax(counts)]

        
        if depth>=self.max_depth or num_labels==1 or num_samples<2*self.min_samples_leaf:
            return Node(label=majority_label)

        feature, treshold, gain= self._best_split(X, y)
        
        if feature is None or gain==0.0:
            return Node(label=majority_label)
        
        left_mask=X[:,feature] <= treshold
        right_mask=X[:,feature] > treshold
        
        left_sub_tree=self._build_tree(X[left_mask,:],y[left_mask] , depth+1)
        right_sub_tree=self._build_tree(X[right_mask,:],y[right_mask] , depth+1)
        
        tree=Node(feature=feature, treshold=treshold, left=left_sub_tree, right=right_sub_tree)

        return tree

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        self.root=self._build_tree(X, y, depth=0)

    def _predict_one(self,x,node):
        if node.label is not None:
            return node.label

        if x[node.feature]<= node.treshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self,X):
        X=np.array(X)
        y=[]        
        for i in range(X.shape[0]):
            y.append(self._predict_one(X[i,:],self.root))
        
        return np.array(y)
    
    def _compute_positions(self, node, depth, x, positions):
        """
        Rekurencyjnie przypisuje współrzędne (x, y) każdemu węzłowi.
        Zwraca aktualny x po przetworzeniu poddrzewa.
        """
        if node is None:
            return x

        x = self._compute_positions(node.left, depth + 1, x, positions)

        positions[node] = (x, -depth)
        x += 1 

        x = self._compute_positions(node.right, depth + 1, x, positions)

        return x

    def _draw_tree(self, node, positions, ax):
        """
        Rekurencyjnie rysuje węzły i krawędzie.
        """
        if node is None:
            return

        x, y = positions[node]

        if node.left is not None:
            x_left, y_left = positions[node.left]
            ax.plot([x, x_left], [y, y_left], color="black")
            self._draw_tree(node.left, positions, ax)

        if node.right is not None:
            x_right, y_right = positions[node.right]
            ax.plot([x, x_right], [y, y_right], color="black")
            self._draw_tree(node.right, positions, ax)

        if node.label is not None:
            # liść
            text = f"label={node.label}"
            facecolor = "#cce5ff"
        else:
            # węzeł wewnętrzny
            text = f"X[{node.feature}] <= {node.treshold:.2f}"
            facecolor = "#ffe5cc"

        circle = plt.Circle((x, y), 0.2, edgecolor="black", facecolor=facecolor)
        ax.add_patch(circle)
        ax.text(x, y, text, ha="center", va="center", fontsize=8)

    def plot_tree(self, figsize=(10, 6)):
        """
        Rysuje drzewo przy użyciu matplotlib.
        """
        import matplotlib.pyplot as plt

        if self.root is None:
            raise ValueError("Drzewo nie jest wytrenowane. Najpierw wywołaj fit().")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()

        positions = {}
        self._compute_positions(self.root, depth=0, x=0, positions=positions)

        self._draw_tree(self.root, positions, ax)

        plt.tight_layout()
        plt.show()


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = load_iris()
X,y=data.data, data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42,test_size=0.3)

my_tree=DecisionTree(max_depth=5,criterion="gini",min_samples_leaf=1)

my_tree.fit(X_train, y_train)

y_hat_my_tree=my_tree.predict(X_test)

sklearn_tree=DecisionTreeClassifier(max_depth=5,criterion="gini",min_samples_leaf=1,random_state=42)

sklearn_tree.fit(X_train,y_train)

y_hat_sklearn=sklearn_tree.predict(X_test)

print("My tree accuracy:",accuracy_score(y_test,y_hat_my_tree))
print("Sklearn tree accuracy:",accuracy_score(y_test,y_hat_sklearn))
print("My tree train acc:", accuracy_score(y_train, my_tree.predict(X_train)))
print("Sklearn train acc:", accuracy_score(y_train, sklearn_tree.predict(X_train)))

print("=== My Tree ===")
print(classification_report(y_test, y_hat_my_tree, digits=3))

print("=== Sklearn Tree ===")
print(classification_report(y_test, y_hat_sklearn, digits=3))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_my_tree=confusion_matrix(y_test,y_hat_my_tree)
cm_sklearn=confusion_matrix(y_test,y_hat_sklearn)

plt.figure(figsize=(5,4))
sns.heatmap(cm_my_tree, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix for my tree")
plt.show()

plt.figure(figsize=(5,4))
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix for sklearn tree")
plt.show()

my_tree.plot_tree()

plt.figure(figsize=(14, 8))
tree.plot_tree(
    sklearn_tree,                      # Twój obiekt DecisionTreeClassifier
    filled=True,                  # kolorowanie wg klasy
    feature_names=data.feature_names,  # opcjonalnie nazwy cech
    class_names=data.target_names,     # opcjonalnie nazwy klas
    rounded=True,                 # zaokrąglone ramki
    fontsize=10
)
plt.title("Decision Tree (scikit-learn)")
plt.show()
    

    
        