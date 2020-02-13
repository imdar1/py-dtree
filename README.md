py-dtree
======
**py-dtree** is a library to build decision tree and predict data using either ID3 or C4.5 algorithm.

Installation
------
### Dependecies:
- python (Tested on python3)
- numpy
- pandas

Currently, pip package is not available. To install this library, you should clone this project manually and add the entire files into your project.
```
git clone https://github.com/imdar1/py-dtree.git
```
Usage
------
Here is some examples of using this library:
1. **Using MyC45**
    ```python
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np
    from myC45 import MyC45

    iris = load_iris()

    data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])

    
    tree = MyC45(gain_ratio=False).fit(data1, iris['feature_names'], 'target', prune=False)
    tree.print_tree()
    ```
    You can set the strategy for selecting an attribute using gain ratio or information gain by modify the `gain_ratio` attribute when MyC45 is instantiated. You are also able to choose whether you want to prune the tree or not by using `prune` argument at `fit` function. 
2. **Using MyID3**
    ```python
    import pandas as pd
    from myID3 import MyID3

    data = pd.read_csv("play_tennis.csv")
    dTree = MyID3(gain_ratio=False)
    dtree_view = dTree.fit(data, ['outlook','temp', 'humidity', 'wind'], 'play')
    dtree_view.print_tree()
    ```
    Like MyC45, you can also set the strategy for selecting an attribute using gain ratio or information gain by modify the `gain_ratio` attribute when MyID3 is instantiated. Note that this algorithm only works for discrete attribute. The algorithm itself is referenced from **Tom Mitchell** book from page 56.

