import os
import clip
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from easyfsl.samplers import TaskSampler
from PIL import Image
import numpy as np
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from statistics import mean
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda")

# 设置随机种子，确保结果的可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义数据增强步骤
augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# 自定义 TensorDataset 类并添加 get_labels 方法
class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors, labels):
        super().__init__(*tensors)
        self.labels = labels
    
    def get_labels(self):
        return self.labels.tolist()
    
    def __getitem__(self, index):
        return self.tensors[0][index], self.labels[index]

# 数据增强扩充样本数量的函数
def augment_sample(image, n_augmentations, transform):
    augmented_images = []
    for _ in range(n_augmentations):
        image_pil = Image.fromarray(np.array(image)).convert('RGB')  # 转换为RGB模式
        augmented_image = transform(image_pil)
        augmented_images.append(augmented_image)
    return augmented_images

# 创建数据集的函数：返回三个数据集
def create_datasets(class_to_indices, my_dataset, total_augmented_images_per_class):
    real_train_images = []
    real_train_labels = []
    augmented_images = []
    augmented_labels = []
    test_images = []
    test_labels = []

    for label, indices in class_to_indices.items():
        selected_indices = random.sample(indices, min(5, len(indices)))
        
        for selected_index in selected_indices:
            image, _ = my_dataset[selected_index]
            real_train_images.append(image)
            real_train_labels.append(label)

        for selected_index in selected_indices:
            image, _ = my_dataset[selected_index]
            image_pil = Image.open(my_dataset.dataset.imgs[selected_index][0])
            augmented_samples = augment_sample(image_pil, n_augmentations=total_augmented_images_per_class - 1, transform=augmentation_transform)
            augmented_images.append(image)
            augmented_images.extend(augmented_samples)
            augmented_labels.extend([label] * total_augmented_images_per_class)

        remaining_indices = [i for i in indices if i not in selected_indices]
        for index in remaining_indices:
            image, _ = my_dataset[index]
            test_images.append(image)
            test_labels.append(label)

    augmented_train_dataset = CustomTensorDataset(torch.stack(augmented_images), labels=torch.tensor(augmented_labels))
    real_train_dataset = CustomTensorDataset(torch.stack(real_train_images), labels=torch.tensor(real_train_labels))
    test_dataset = CustomTensorDataset(torch.stack(test_images), labels=torch.tensor(test_labels))

    return augmented_train_dataset, real_train_dataset, test_dataset

# CLIP + Adapter 类定义，增加文本原型部分
class CLIPWithAdapter(nn.Module):
    def __init__(self, clip_model, num_classes, class_names):
        super().__init__()
        self.clip_visual = clip_model.visual  # CLIP 模型的视觉编码器
        self.clip_text = clip_model.encode_text  # CLIP 模型的文本编码器
        self.num_classes = num_classes
        self.class_names = class_names  # 类别名称
        self.clip_output_dim = self.clip_visual.output_dim  # CLIP 输出的特征维度
        
        # 定义 Adapter 层，输入是拼接后的特征维度（CLIP 输出 + 类别独热编码长度），输出维持 CLIP 的输出维度
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_output_dim + num_classes, self.clip_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 预先计算文本原型
        self.text_prototypes = self.compute_text_prototypes()

    def compute_text_prototypes(self):
        """
        计算文本原型
        """
        text_inputs = [f"a picture of {class_name}" for class_name in self.class_names]
        # text_inputs = [f"{class_name}" for class_name in self.class_names]

        text_tokens = clip.tokenize(text_inputs).to(DEVICE)
        with torch.no_grad():
            text_features = self.clip_text(text_tokens)
        return text_features

    def compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        计算图像原型，并与文本原型加权求和，返回多模态原型。
        """
        class_prototypes = []
        for class_label in range(self.num_classes):
            # 使用mask选择每类样本的特征
            class_mask = (support_labels == class_label)
            class_features = support_features[class_mask]
            if len(class_features) == 0:
                continue  # 跳过没有样本的类别
            image_prototype = class_features.mean(dim=0)
            text_prototype = self.text_prototypes[class_label]
            # 图像和文本原型加权求和（确保同类别相加）
            multi_modal_prototype = 0.7 * image_prototype + 0.3 * text_prototype
            class_prototypes.append(multi_modal_prototype)
        
        return torch.stack(class_prototypes)
    
    
    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor):
        """
        前向传播：先通过 CLIP 提取图像和文本原型，然后多模态原型拼接独热编码送入 Adapter 调整并计算距离。
        """
        with torch.no_grad():  # 冻结 CLIP 主干
            # 提取支持集和查询集的特征
            support_features = self.clip_visual(support_images).float()
            query_features = self.clip_visual(query_images).float()

        # 计算多模态原型（未拼接独热编码）
        prototypes = self.compute_prototypes(support_features, support_labels)

        # 先对每个类别的原型拼接独热编码，并通过 Adapter
        final_prototypes = []
        for class_label in range(self.num_classes):
            one_hot = torch.zeros(self.num_classes, device=support_labels.device)
            one_hot[class_label] = 1
            if class_label < len(prototypes):  # 如果该类存在原型
                prototype = prototypes[class_label]
                prototype = torch.cat([prototype, one_hot], dim=0)  # 拼接独热编码
                prototype = self.adapter(prototype.unsqueeze(0)).squeeze(0)  # 通过 Adapter
                final_prototypes.append(prototype)

        # 将最终的原型组合成一个 tensor
        final_prototypes = torch.stack(final_prototypes)

        # 对查询集特征拼接零向量后送入 Adapter
        query_features = torch.cat([query_features, torch.zeros(query_features.size(0), self.num_classes).to(query_features.device)], dim=1)
        query_features = self.adapter(query_features)

        # 计算查询集和支持集原型之间的距离
        distances = torch.cdist(query_features, final_prototypes)
        
        return distances

# 训练过程
def training_epoch(model: CLIPWithAdapter, data_loader: DataLoader, optimizer: torch.optim.Optimizer):
    all_loss = []
    model.train()
    LOSS_FUNCTION = nn.CrossEntropyLoss()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            optimizer.zero_grad()
            distances = model(support_images.to(DEVICE), support_labels.to(DEVICE), query_images.to(DEVICE))
            negative_distances = -distances
            loss = LOSS_FUNCTION(negative_distances, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss))
    return mean(all_loss)

# 从真实训练集计算类别原型
def compute_prototypes_from_real_trainset(model: CLIPWithAdapter, real_train_loader: DataLoader, device: str = "cuda") -> torch.Tensor:
    """
    计算多模态原型，确保图像原型和文本原型加权时类别是对应的，
    然后拼接独热编码进入 Adapter,得到最后的多模态原型。
    """
    model.eval()
    class_features = {}
    class_counts = {}
    
    # 统计每个类别的图像特征
    with torch.no_grad():
        for images, labels in real_train_loader:
            images, labels = images.to(device), labels.to(device)

            # 提取图像特征
            support_features = model.clip_visual(images).float()

            # 将特征按类别累加
            for feature, label in zip(support_features, labels):
                label = label.item()  # 获取类别索引
                if label not in class_features:
                    class_features[label] = feature
                    class_counts[label] = 1
                else:
                    class_features[label] += feature
                    class_counts[label] += 1

    # 计算每类的图像原型，并与对应的文本原型加权求和，最终通过 Adapter 层调整
    prototypes = []
    for class_label in range(model.num_classes):
        # 确保每个类别都有样本
        assert class_label in class_features, f"类别 {class_label} 没有足够的样本进行训练"

        # 图像原型为每类特征的均值
        image_prototype = class_features[class_label] / class_counts[class_label]
        text_prototype = model.text_prototypes[class_label]

        # 对应类别的图像和文本原型加权求和
        multi_modal_prototype = 0.7 * image_prototype + 0.3 * text_prototype

        # 将多模态原型与类别的独热编码拼接并通过 Adapter
        one_hot = torch.zeros(model.num_classes, device=device)
        one_hot[class_label] = 1
        multi_modal_prototype = torch.cat([multi_modal_prototype, one_hot], dim=0)
        final_prototype = model.adapter(multi_modal_prototype.unsqueeze(0)).squeeze(0)
        prototypes.append(final_prototype)

    return torch.stack(prototypes).to(device)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(features, labels, epoch, accuracy, save_path, dpi=1000):
    """
    绘制TSNE图像,并将其保存为文件。
    :param features: 特征矩阵
    :param labels: 每个特征对应的标签
    :param epoch: 当前训练的epoch数
    :param accuracy: 当前epoch的准确率
    :param save_path: 保存图像的路径
    :param dpi: 图像分辨率
    """
    # 使用TSNE降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    
    # 调整 s 的大小，让点更大
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', s=40)
    
    # 修改颜色条，使其显示 1-10 而不是 0-9
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Classes', fontname='Times New Roman')
    cbar.set_ticks(range(10))  # 设置刻度为 0-9
    cbar.set_ticklabels(range(1, 11))  # 设置标签为 1-10
    
    # 保存图像
    file_name = f"TSNE_epoch{epoch}_accuracy_{accuracy:.2f}.png"
    plt.savefig(f"{save_path}/{file_name}", dpi=dpi)
    plt.close()


# 评估集评估
def evaluate_with_train_prototypes_and_tsne(
    model: CLIPWithAdapter, 
    test_loader: DataLoader, 
    prototypes: torch.Tensor, 
    epoch: int, 
    device: str = "cuda",
    log_file: str = None,
    save_path: str = "./2.0_Reuslt"
):
    total_predictions = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    all_features = []

    class_correct = {}
    class_total = {}

    model.eval()
    with torch.no_grad():
        for query_images, query_labels in tqdm(test_loader, desc="Evaluating"):
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            query_features = model.clip_visual(query_images).float()
            query_features = torch.cat([query_features, torch.zeros(query_features.size(0), model.num_classes).to(query_features.device)], dim=1)
            query_features = model.adapter(query_features)

            distances = torch.cdist(query_features, prototypes)
            predictions = distances.argmin(dim=1)
            
            correct = (predictions == query_labels).sum().item()
            total_predictions += len(query_labels)
            correct_predictions += correct
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
            all_features.extend(query_features.cpu().numpy())  # 保存所有特征用于TSNE

            for label, prediction in zip(query_labels.cpu().numpy(), predictions.cpu().numpy()):
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

    overall_accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average='macro') * 100
    recall = recall_score(all_labels, all_predictions, average='macro') * 100
    f1 = f1_score(all_labels, all_predictions, average='macro') * 100

    with open(log_file, "a") as f:
        for label in class_total:
            accuracy = 100 * class_correct[label] / class_total[label]
            print(f"Class {label+1}: accuracy {accuracy:.2f}%")
            f.write(f"Class {label+1}: accuracy {accuracy:.2f}%\n")
        f.write("\n")
        f.write(f"overall_accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"precision: {precision:.2f}%\n")
        f.write(f"recall: {recall:.2f}%\n")
        f.write(f"F1-Score: {f1:.2f}%\n")
        f.write("======================================================\n")
    
    print("\n")
    print(f"overall_accuracy: {overall_accuracy:.2f}%")
    print(f"precision: {precision:.2f}%")
    print(f"recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")


    # 绘制并保存TSNE图
    plot_tsne(
        features=np.array(all_features),
        labels=np.array(all_labels),
        epoch=epoch,
        accuracy=overall_accuracy,
        save_path=save_path,
        dpi=1000
    )

    return overall_accuracy


seeds = [42]


# 类别名称

class_names = ['Alternaria leaf spot', 'Apple scab', 'Black rot', 'Brown spot', 'Cedar apple rust', 'Frogeye leaf spot', 'Grey spot', 'Healthy', 'mosaic virus', 'Powdery mildew']

# class_names = [
#     "This is a picture of a Alternaria leaf spot: Irregular brown/yellow spots, darker edges, black mold on underside, irregular lesion lesions.",
#     "This is a picture of a Scab apple leaves: Gray spots, indistinct edges, dark gray veins, variable size/shape, scarring.",
#     "This is a picture of a Black rot: Dark, sunken lesions on leaves and fruit, can cause premature leaf drop, often accompanied by a blackened stem.",
#     "This is a picture of a Brown spot leaves: Abundant yellowing leaves, brown/black lesions, noticeable edge damage, variable lesion size, leaf health compromised.",
#     "This is a picture of a Rust apple leaves: Scattered orange-yellow/black spots, varied size/shape, slightly elevated.",
#     "This is a picture of a Frogeye leaf spot: Round, dark lesions with a light gray center, often surrounded by a yellow halo, leads to early leaf drop.",
#     "This is a picture of a Grey spot leaves: Green leaves with irregular gray-brown spots, clear margins, speckled, mildew presence indicated.",
#     "This is a picture of a Healthy apples leaves: Vibrant green color, smooth texture, no disease spots, glossy, signs of optimal health.",
#     "This is a picture of a Mosaic virus: Mottled yellow/green pattern, uneven color, no distinct spots, leaf pattern disrupted, can lead to reduced yield.",
#     "This is a picture of a Powdery mildew leaves: White/gray powder, uneven color, full leaf coating, fuzzy."
# ]


# 创建数据集
root_dir = './dataset2'
my_dataset = MyDataset(root_dir, transform=transform)
class_to_indices = {}
for idx, (img_path, label) in enumerate(my_dataset.dataset.imgs):
    if label not in class_to_indices:
        class_to_indices[label] = []
    class_to_indices[label].append(idx)

N_WAY = 10
N_SHOT = 1
N_QUERY = 19
N_TRAIN_TASKS = 10
total_augmented_images_per_class = 100
n_epochs = 50

# 循环执行训练过程，每次使用不同的种子
for seed in seeds:

    # 设置随机种子，确保结果的可复现性
    set_seed(seed)

    # 加载 CLIP 模型
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    clip_model = clip_model.float()

    # 重新创建数据集
    augmented_train_dataset, real_train_dataset, test_dataset = create_datasets(class_to_indices, my_dataset, total_augmented_images_per_class)

    train_sampler = TaskSampler(augmented_train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAIN_TASKS)
    train_loader = DataLoader(augmented_train_dataset, batch_sampler=train_sampler, pin_memory=True, collate_fn=train_sampler.episodic_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    real_train_loader = DataLoader(real_train_dataset, batch_size=len(real_train_dataset), shuffle=False)

    # 初始化模型
    few_shot_classifier = CLIPWithAdapter(clip_model, num_classes=10, class_names=class_names).to(DEVICE)

    # 优化器只更新 Adapter 的参数
    train_optimizer = Adam(few_shot_classifier.adapter.parameters(), lr=0.01, weight_decay=5e-4)
    train_scheduler = MultiStepLR(train_optimizer, milestones=[29, 50], gamma=0.1)

    # 创建一个基于当前种子的日志文件，并保存在 seed 文件夹中
    os.makedirs("2.0_Result", exist_ok=True)
    log_file = f"./2.0_Result/2.0_training_log.txt"

    # 训练过程
    with open(log_file, "w") as f:
        f.write(f"2.0_Training and Evaluation Log\n")
        f.write("======================================================\n")

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        train_scheduler.step()

        # 使用训练集的原型进行测试集评估，并传入日志文件路径
        prototypes = compute_prototypes_from_real_trainset(few_shot_classifier, real_train_loader, device=DEVICE)
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{n_epochs}\n")
            f.write(f"Average Loss: {average_loss:.4f}\n")

        evaluate_with_train_prototypes_and_tsne(few_shot_classifier, test_loader, prototypes, epoch=epoch + 1, device=DEVICE, log_file=log_file, save_path="./2.0_Result")


    print(f"Log saved to {log_file}")
