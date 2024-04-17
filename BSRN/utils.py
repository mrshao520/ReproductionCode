import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd

def make_grid(num_layer, outputs, tag=''):
    size = outputs.size(0)
    col = size // 8 + 1
    
    plt.figure(figsize=(100, 100))
    for i, image in enumerate(outputs):
        plt.subplot(col, 8, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"D:/home/ReproductionCode/BSRN/results/layer_{num_layer}_{tag}.png")
    # plt.show()
    plt.close()
    
def calculate_relation(num_layer, outputs, tag=''):
    size = outputs.size(0)
    results = []
    for i in range(size):
        for j in range(i + 1, size):
            distance = ((outputs[i] - outputs[j]).pow(2).sum() / (outputs.size(1) * outputs.size(2)))
            results.append(distance.item())
    
    data = pd.Series(results)
    plt.hist(x=data, bins=1000)
    data.plot(kind='kde')
    plt.savefig(f"D:/home/ReproductionCode/BSRN/results/layer_{num_layer}_{tag}.png")
    plt.close()
    
def viaualize_feature(num_layer, outputs, tag=''):
    make_grid(num_layer, outputs, tag + '_feature')
    calculate_relation(num_layer, outputs, tag + '_relation')
            
if __name__ == '__main__':
    outputs = torch.randn((10, 2, 2))
    print(outputs)
    calculate_relation(2, outputs)