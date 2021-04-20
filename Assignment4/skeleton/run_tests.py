
if __name__ == '__main__':
    allgood = True
    try:
        import numpy as np
        A = np.ones((5,5))
        print('`import numpy` ok!')
    except:
        allgood = False
        print('`import numpy` failed')

    try:
        import torch
        A = torch.randn(5,5)
        print('`import torch` ok!')
    except:
        allgood = False
        print('`import torch` failed')

    try:
        import torchvision
        model=torchvision.models.segmentation.fcn_resnet101()
        print('`import torchvision` ok!')
    except:
        allgood = False
        print('`import torchvision` failed')

    try:
        import matplotlib.pyplot as plt
        h = plt.figure()
        plt.close(h)
        print('`import matplotlib.pyplot` ok!')
    except:
        allgood = False
        print('`import matplotlib.pyplot` failed')


    if allgood:
        print("\n\nEverything is ok!")
