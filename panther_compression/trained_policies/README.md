1. test1.pt: 256x256 pos_loss:1e-3, yaw_loss: 4e-3, some of the pos trajs are not stable, but its best trajectory is pretty good
2. test2.pt: 256x256 pos_loss:3e-4, yaw_loss: 1e-3, make yaw_loss_weight 10 -> 5, still see yaw_dot violation
3. test3.pt: 512x512 pos_loss:2e-4, yaw_loss: 4e-4, 
4. test4.pt: 512x512x512 pos_loss:1e-4, 1.5e-4, trained for simulations (accidentally deleted)
5. test5.pt: 512x512x512 pos_loss:1e-4, 1.6e-4, trained for hardware
6. test6.pt: 1024x1024x1024 pos_loss:1e-4, 1.2e-4, trained for hardware
7. test7.pt: 1024x1024x1024 pos_loss:7e-5, 1e-4, trained for hardware. same with test6.pt but trained longer
8. 
