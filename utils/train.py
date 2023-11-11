from itertools import cycle
dataloader2 = cycle(dataloader2)
for data1, data2 in zip(dataloader1, dataloader2):
    # 处理 data1 和 data2
