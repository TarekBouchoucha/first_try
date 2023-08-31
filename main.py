import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
import time
import os
from random import sample
import matplotlib.pyplot as plt
import torchvision.models as models

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.data = []
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.data.append((image_path, i))  # (image_path, class_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")  # Make sure the images are RGB format
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [label for _, label in self.data]

image_size = 50

# Custom dataset root path
data_root = "C:/Users/ASUS/Desktop/train"

# Data transformations for train and test sets
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Custom dataset for training and testing
train_set = CustomDataset(root=data_root, transform=train_transform)
test_set = CustomDataset(root=data_root, transform=test_transform)

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


convolutional_network = resnet18(weights=True)
convolutional_network.fc = nn.Flatten()

model = PrototypicalNetworks(convolutional_network)

N_WAY = 20  # Number of classes in a task (bmw, acura, audi)
N_SHOT = 20  # Number of images per class in the support set
N_QUERY = 5  # Number of images per class in the query set
N_EVALUATION_TASKS = 100  # Number of evaluation tasks

# Custom function to randomly select support and query images for each task
def get_random_support_and_query_indices(class_indices, n_shot, n_query):
    support_indices = []
    query_indices = []
    for class_index in class_indices:
        all_indices = [i for i, (_, label) in enumerate(train_set.data) if label == class_index]
        support_indices.extend(sample(all_indices, n_shot))
        query_indices.extend(sample(all_indices, n_query))
    return support_indices, query_indices

# Create a custom sampler for 100 tasks with 100 shots and 1 query per class
from easyfsl.samplers import TaskSampler

class ThreeWaySampler(TaskSampler):
    def __init__(self, dataset, n_shot, n_query, n_tasks):
        super(ThreeWaySampler, self).__init__(
            dataset, n_way=N_WAY, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
        )

    def get_support_query_indices(self):
        classes = list(range(len(self.dataset.classes)))
        support_indices, query_indices = get_random_support_and_query_indices(
            classes, self.n_shot, self.n_query
        )
        return support_indices, query_indices
def plot_images(images, title, images_per_row=5):
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    plt.figure(figsize=(10, 10))
    for i, image in enumerate(images):
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(image.permute(1, 2, 0))  # Convert tensor to image format (C, H, W) -> (H, W, C)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

test_sampler = ThreeWaySampler(
    test_set, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=0,
    pin_memory=False,
    collate_fn=test_sampler.episodic_collate_fn,
)
print('hello')

(
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
) = next(iter(test_loader))

plot_images(example_support_images, "support images", images_per_row=N_SHOT)
plot_images(example_query_images, "query images", images_per_row=N_QUERY)

model.eval()
example_scores = model(
    example_support_images,
    example_support_labels,
    example_query_images,
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)

print("Ground Truth / Predicted")
for i in range(len(example_query_labels)):
    true_class_name = test_set.classes[example_query_labels[i]]
    predicted_class_name = test_set.classes[example_predicted_labels[i]]
    print(f"{true_class_name} / {predicted_class_name}")

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images, support_labels, query_images)
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    # Start the timer
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")
    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )


evaluate(test_loader)

N_TRAINING_EPISODES = 400
N_VALIDATION_TASKS = 100

# Remove the lambda function for train_set.get_labels since it is already defined in the CustomDataset class

train_sampler = ThreeWaySampler(
    train_set, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)

train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=False,
    collate_fn=train_sampler.episodic_collate_fn,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def sliding_average(values, window):
    if len(values) < window:
        return sum(values) / len(values)
    return sum(values[-window:]) / window
def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images, support_labels, query_images
    )

    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

print("done")

# Train the model yourself with this cell

log_update_frequency = 10

all_loss = []
# Start the timer
start_time = time.time()

model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)

        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

# End the timer
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

evaluate(test_loader)

torch.save(model.state_dict(), 'few_shot_learning_model_all_classes.pth')
