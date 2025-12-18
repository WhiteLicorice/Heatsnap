# Status: Plateau at around ~10% degrees of error. Need to break through to at least 5% to be publishable.

---

# Vision Only

main: using device cuda

Epoch 1/20
main: train_loss (Huber) is 23.4906 | val_loss (Huber) is 17.2981
main: validation loss improved, saved to checkpoints\best_model.pth

Epoch 2/20
main: train_loss (Huber) is 7.5300 | val_loss (Huber) is 15.9892
main: validation loss improved, saved to checkpoints\best_model.pth

Epoch 3/20
main: train_loss (Huber) is 6.4463 | val_loss (Huber) is 15.2254                                                             
main: validation loss improved, saved to checkpoints\best_model.pth

Epoch 4/20
main: train_loss (Huber) is 6.0250 | val_loss (Huber) is 14.8644                                                             
main: validation loss improved, saved to checkpoints\best_model.pth

Epoch 5/20
main: train_loss (Huber) is 5.7311 | val_loss (Huber) is 14.4409                                                             
main: validation loss improved, saved to checkpoints\best_model.pth

Epoch 6/20
main: train_loss (Huber) is 5.3825 | val_loss (Huber) is 15.0198                                                             

Epoch 7/20
main: train_loss (Huber) is 5.0889 | val_loss (Huber) is 15.9081                                                             

Epoch 8/20
Epoch 00008: reducing learning rate of group 0 to 1.0000e-05.                                                                
main: train_loss (Huber) is 4.8142 | val_loss (Huber) is 15.2314

Epoch 9/20
main: train_loss (Huber) is 4.1443 | val_loss (Huber) is 16.0530

---

# Vision and Physics Layer

(venv) PS D:\Creative Corner\Projects\Software\Heatsnap> python train.py
main: using device cuda
main: training for 20 epochs...
Epoch 01 | Train: 11.9363 | Val: 14.5788 | 988.8s
main: validation loss improved, saved to checkpoints\best_model.pth
Epoch 02 | Train: 6.1929 | Val: 12.7124 | 1036.1s
main: validation loss improved, saved to checkpoints\best_model.pth
Epoch 03 | Train: 5.7262 | Val: 11.6369 | 1030.2s
main: validation loss improved, saved to checkpoints\best_model.pth
Epoch 04 | Train: 5.3977 | Val: 12.3619 | 1034.6s