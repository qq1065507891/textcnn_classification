
import fasttext
train_path = "./train.txt"
test_path = "./test.txt"
wf_model = fasttext.train_supervised(train_path, epoch=20, wordNgrams=2, minCount=5)
wf_model.save_model("./wf_model")
wf_model = fasttext.load_model("./wf_model")
input = []
target = []
for line in open(test_path, encoding='utf-8').readlines():
    temp = line.split("__label__")
    input.append(temp[0].strip())
    target.append(temp[1].strip())
# 使用特征和模型进行预测
labels, acc_list = wf_model.predict(input)
# 计算准确率
sum = 0
print(len(labels), len(target))
for i, j in zip(labels, target):
    if i[0].replace("__label__", "") == j:
        sum += 1
acc = sum / len(labels)  # 平均的准确率
print(acc)