1. test1.pt: 256x256 pos_loss:1e-3, yaw_loss: 4e-3, some of the pos trajs are not stable, but its best trajectory is pretty good
2. test2.pt: 256x256 pos_loss:3e-4, yaw_loss: 1e-3, make yaw_loss_weight 10 -> 5, still see yaw_dot violation
3. test3.pt: 512x512 pos_loss:2e-4, yaw_loss: 4e-4, 
4. test4.pt: 512x512x512 pos_loss:1e-4, 1.5e-4
