import decision_tree
import json
import tree_plotter

fr = open(r'/home/zhaoguanyi/PycharmProjects/Decision Tree/watermelon.txt')

listWm = [inst.strip().split('\t') for inst in fr.readlines()]  # 读取数据集
labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  # 标签
Trees = decision_tree.createTree(listWm, labels)  # 构建决策树

print(json.dumps(Trees, ensure_ascii=False))  # 打印决策树

labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
testData = ['浅白', '蜷缩', '浊响', '稍糊', '凹陷', '硬滑']  # 测试数据
testClass = decision_tree.classify(Trees, labels, testData)  # 测试
print(json.dumps(testClass, ensure_ascii=False))

tree_plotter.createPlot(Trees)  # 可视化决策树
