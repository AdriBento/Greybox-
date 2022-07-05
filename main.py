import network
import utils
import os
import random
import numpy as np
from torch.utils import data
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from validation import validate
from load_dataset import get_dataset
from parameters import get_arguments
from utils import apply_confidence_mask, make_hot_deeplab, enet_weighing


def main():
    # Set up parameters
    opts = get_arguments().parse_args()
    confidence_threshold = opts.deeplab_confidence_threshold
    num_classes = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up metrics
    metrics = StreamSegMetrics(num_classes)
    correct_metrics = StreamSegMetrics(num_classes)
    wrong_metrics = StreamSegMetrics(num_classes)

    # Load dataset
    train_dst, val_dst, test_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    # Set up Latent Space Predictor (lsp)
    latent_space_predictor = network.deeplabv3plus_resnet101(num_classes=num_classes, output_stride=opts.output_stride)
    network.convert_to_separable_conv(latent_space_predictor.classifier)
    utils.set_bn_momentum(latent_space_predictor.backbone, momentum=0.01)

    # Set up Transparent Classifier (tc)
    transparent_classifier = network.LinearRegression()

    # Set up optimizers and schedulers
    optimizer_lsp = torch.optim.SGD(params=[
        {'params': latent_space_predictor.backbone.parameters(), 'lr': 0.1 * opts.lr_deeplab},
        {'params': latent_space_predictor.classifier.parameters(), 'lr': opts.lr_deeplab}
    ], lr=opts.lr_deeplab, momentum=0.9, weight_decay=opts.weight_decay)

    optimizer_tc = torch.optim.SGD(params=[
        {'params': transparent_classifier.parameters(), 'lr': opts.lr_logreg},
    ], lr=opts.lr_logreg, momentum=0.9, weight_decay=1e-4)

    scheduler_deeplab = torch.optim.lr_scheduler.StepLR(optimizer_lsp, step_size=opts.step_size, gamma=0.1)
    scheduler_logreg = torch.optim.lr_scheduler.StepLR(optimizer_tc, step_size=10000, gamma=0.1)

    # Set up class weights
    weights = enet_weighing(train_loader, num_classes)
    class_weights = torch.FloatTensor(weights).cuda()  # Convert to Tensor

    # Set up criterion
    criterion_deeplab = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')
    criterion_logreg = nn.CrossEntropyLoss(reduction='mean')

    # Prepare variables
    best_itrs = 0.0
    best_score = 0.0
    associated_test_score = 0.0
    cur_itrs, cur_epochs = 0, 0
    interval_loss_total = 0
    interval_loss_logreg = 0
    interval_loss_deeplab = 0
    prev_loss = 0

    # Prepare to save results
    datalog = {'Val Accuracy': [], 'Test Accuracy': [], 'Learning Rate': [], 'Loss': [], 'Iterations': []}

    # Send models to GPU
    latent_space_predictor = nn.DataParallel(latent_space_predictor)
    latent_space_predictor.to(device)
    transparent_classifier = transparent_classifier.to(device)

    # Load pre-train LSP and TC
    deeplab_loaded = False  # could be a parameter
    print("Deeplab is loaded", deeplab_loaded)
    if deeplab_loaded:
        checkpoint = torch.load('checkpoints/deeplab_ready.pth')
        latent_space_predictor.load_state_dict(checkpoint['model_state_dict'])
        optimizer_lsp.load_state_dict(checkpoint['optimizer_state_dict'])

    logreg_loaded = True  # could be a parameter
    print("Logreg is loaded", logreg_loaded)
    if logreg_loaded:
        checkpoint = torch.load('checkpoints/linreg_classifier_v2.pth')
        transparent_classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer_tc.load_state_dict(checkpoint['optimizer_state_dict'])

    m = nn.Softmax2d()

    # Decide what to train
    train_deeplab = False  # could be a parameter
    train_logreg = False  # could be a parameter
    train_deeplab_for_logreg = True  # could be a parameter

    # Test only
    if opts.test_only:
        latent_space_predictor.eval()
        transparent_classifier.eval()
        acc, val_score = validate(opts=opts, latent_space_predictor=latent_space_predictor,
                                  transparent_classifier=transparent_classifier, loader=val_loader,
                                  device=device, metrics=metrics,
                                  correct_metrics=correct_metrics, wrong_metrics=wrong_metrics)
        print(metrics.to_str(val_score))
        return

    while True:
        # =====  Train Loop ===== #
        cur_epochs += 1
        # Start training
        for (name, images, labels, labels_class) in train_loader:
            cur_itrs += 1

            # Load images and labels
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_class = labels_class.to(device, dtype=torch.long)

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

            # Apply losses
            loss_logreg = criterion_logreg(pred_classes, labels_class)
            loss_deeplab = criterion_deeplab(pred_attributes, labels)

            # Trade-off between Deeplab et Logreg losses
            beta = 0
            gamma = 1
            loss_deeplab_for_logreg_opti = beta * loss_logreg + gamma * loss_deeplab

            # Train logreg alone
            if train_logreg:
                optimizer_tc.zero_grad()
                loss_logreg.backward()
                optimizer_tc.step()
                np_loss_logreg = loss_logreg.detach().cpu().numpy()
                interval_loss_logreg += np_loss_logreg
                interval_loss_total += np_loss_logreg

            # Train Deeplab alone
            if train_deeplab:
                optimizer_lsp.zero_grad()
                loss_deeplab.backward()
                optimizer_lsp.step()
                np_loss_deeplab = loss_deeplab.detach().cpu().numpy()
                interval_loss_deeplab += np_loss_deeplab

            # Train both Deeplab and Logreg (but only Deeplab if beta = 0 and gamma = 1)
            if train_deeplab_for_logreg:
                optimizer_lsp.zero_grad()
                loss_deeplab_for_logreg_opti.backward()
                optimizer_lsp.step()
                np_loss_deeplab = loss_deeplab_for_logreg_opti.detach().cpu().numpy()
                interval_loss_deeplab += np_loss_deeplab

            # Print current state of training
            if cur_itrs % 10 == 0:
                interval_loss_deeplab = interval_loss_deeplab / 10
                interval_loss_logreg = interval_loss_logreg / 10
                interval_loss_total = interval_loss_total / 10
                print("Epoch %d, Itrs %d/%d, Loss DeepLab=%f, Loss MLP=%f, Total Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss_deeplab, interval_loss_logreg,
                       interval_loss_total))

                prev_loss = interval_loss_logreg

                interval_loss_logreg = 0.0
                interval_loss_total = 0.0
                interval_loss_deeplab = 0.0

            # Proceed to evaluation
            if cur_itrs % opts.val_interval == 0:
                latent_space_predictor.eval()
                transparent_classifier.eval()
                acc, val_score = validate(opts=opts, latent_space_predictor=latent_space_predictor,
                                          transparent_classifier=transparent_classifier,
                                          loader=val_loader, device=device, metrics=metrics,
                                          correct_metrics=correct_metrics, wrong_metrics=wrong_metrics)
                acc_test, test_score = validate(opts=opts, latent_space_predictor=latent_space_predictor,
                                                transparent_classifier=transparent_classifier,
                                                loader=test_loader, device=device, metrics=metrics,
                                                correct_metrics=correct_metrics, wrong_metrics=wrong_metrics)

                # Store data for analysis
                datalog['Val Accuracy'].append(acc)
                datalog['Test Accuracy'].append(acc_test)
                datalog["Learning Rate"].append((optimizer_tc.param_groups[0]['lr']))
                datalog['Loss'].append(prev_loss)
                datalog['Iterations'].append(cur_itrs)

                # Store best score
                if acc > best_score:
                    best_score = acc
                    best_itrs = cur_itrs
                    print("THIS IS THE BEST SCORE")

                # End eval
                latent_space_predictor.train()
                transparent_classifier.train()

            # Step schedulers
            if train_logreg:
                scheduler_logreg.step()

            if train_deeplab:
                scheduler_deeplab.step()

            if train_deeplab_for_logreg:
                scheduler_logreg.step()
                scheduler_deeplab.step()

            # Print of the final results and plot
            if cur_itrs >= opts.total_itrs:
                print("Best result was", best_score, "and it happened at Itrs", best_itrs)
                print("Associated test score is", associated_test_score)

                df = pd.DataFrame.from_dict(datalog, orient="index").transpose()
                df.to_csv('jsons/best_linreg_classifier_monumai.csv')
                figure, axes = plt.subplots(1, 3)
                df.plot(x='Iterations', y=['Test Accuracy'], kind='line', xlabel="Iterations", ax=axes[0])
                df.plot(x='Iterations', y=['Val Accuracy'], kind='line', xlabel="Iterations", ax=axes[0])
                df.plot(x='Iterations', y=['Loss'], kind='line', xlabel="Iterations", ax=axes[1])
                df.plot(x='Iterations', y=['Learning Rate'], kind='line', xlabel="Iterations", ax=axes[2])
                plt.show()
                return


if __name__ == '__main__':
    main()
