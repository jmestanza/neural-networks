test_csv = pd.read_csv("cifar100_test.csv")
train_csv =pd.read_csv("cifar100_train.csv")

num_to_label = {}
x_train_labels = train_csv["Label"]
for i in range(len(y_train)):
    if not int(y_train[i]) in num_to_label:
        num_to_label[int(y_train[i])] = x_train_labels[i]

#print(num_to_label)

label_col = np.zeros(len(test_csv)).astype(str)
for i in range(len(label_col)):
    num = int(y_test[i])
    label_col[i] = num_to_label[num] 
test_csv["Label"] = label_col

if not os.path.exists('cifar100_test_posta.csv'):
    test_csv.to_csv('cifar100_test_posta.csv', index=False)