from tqdm import tqdm
import torch
import torch.nn as nn
from utils import apply_confidence_mask, make_hot_deeplab


def validate(opts, latent_space_predictor, transparent_classifier, loader, device, metrics, correct_metrics,
             wrong_metrics):
    """
    Do validation and return specified samples
    """
    # Reset metrics
    metrics.reset()
    correct_metrics.reset()
    wrong_metrics.reset()

    loader_name = str(loader)
    confidence_threshold = opts.deeplab_confidence_threshold
    m = nn.Softmax2d()

    # Initialize variables
    list_correct_preds, list_correct_targets, list_wrong_preds, list_wrong_targets = [], [], [], []
    correct_teacher_accuracy, correct_teacher_gt_accuracy, correct_classes, \
        total_classes, correct, correct_target, correct_preds = 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for i, (name, images, labels, labels_class) in tqdm(enumerate(loader)):
            # Load images and labels
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_class = labels_class.to(device, dtype=torch.long)
            targets_attributes = labels.cpu().numpy()

            # Use Deeplab to predict the attributes
            pred_attributes = latent_space_predictor(images)
            softmax_attributes = m(pred_attributes)

            # Apply confidence mask
            confident_pred_attributes = apply_confidence_mask(pred_attributes, softmax_attributes, confidence_threshold)

            # Transform segmentation mask into one hot vector and put it on GPU as a tensor
            one_hot_predicted_attrib = make_hot_deeplab(images, confident_pred_attributes)
            one_hot_predicted_attrib_tensor = torch.tensor(one_hot_predicted_attrib)
            one_hot_predicted_attrib_tensor = one_hot_predicted_attrib_tensor.to(device, dtype=torch.long)

            # Use the Transparent Classifier on the list of predicted attribute in order to predict classes
            pred_classes = transparent_classifier(one_hot_predicted_attrib_tensor.float())
            max_preds_classes = pred_classes.detach().max(dim=1)[1]

            # Verify if those classes are predicted correctly
            correct_classes += max_preds_classes.eq(labels_class.data).cpu().sum().float().item()
            total_classes += pred_classes.size(0)

            # Split images in 2 stacks : those with correctly predicted classes and the others
            pred_attributes_cpu = pred_attributes.detach().max(dim=1)[1].cpu().numpy()
            for j in range(0, len(max_preds_classes)):
                if max_preds_classes[j] == labels_class[j]:
                    list_correct_preds.append(pred_attributes_cpu[j])
                    list_correct_targets.append(targets_attributes[j])
                else:
                    list_wrong_preds.append(pred_attributes_cpu[j])
                    list_wrong_targets.append(targets_attributes[j])

            # Calculate IOU Metric over the 2 different stacks
            metrics.update(targets_attributes, pred_attributes_cpu)
            correct_metrics.update(list_correct_targets, list_correct_preds)
            wrong_metrics.update(list_wrong_targets, list_wrong_preds)

        score = metrics.get_results()
        # score_correct = correct_metrics.get_results()
        # score_wrong = wrong_metrics.get_results()
        acc_logreg = correct_classes / total_classes

        print("####################################")
        print('Accuracy LogReg based on detected attributes on', loader_name, "is", acc_logreg)
        print(metrics.to_str(score))

    return acc_logreg, score
