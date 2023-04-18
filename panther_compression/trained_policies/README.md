1. test1.pt: 256x256 pos_loss:1e-3, yaw_loss: 4e-3, some of the pos trajs are not stable, but its best trajectory is pretty good
2. test2.pt: 256x256 pos_loss:3e-4, yaw_loss: 1e-3, make yaw_loss_weight 10 -> 5, still see yaw_dot violation
3. test3.pt: 512x512 pos_loss:2e-4, yaw_loss: 4e-4, 
4. test4.pt: 512x512x512 pos_loss:1e-4, 1.5e-4, trained for simulations (accidentally deleted)
5. test5.pt: 512x512x512 pos_loss:1e-4, 1.6e-4, trained for hardware
6. test6.pt: 1024x1024x1024 pos_loss:1e-4, 1.2e-4, trained for hardware
7. test7.pt: 1024x1024x1024 pos_loss:7e-5, 1e-4, trained for hardware. same with test6.pt but trained longer
8. test8.pt: 1024x1024x1024x1024 pos_loss: 7e-5, 8e-5, tanh layer at the end, trained for hardware, evaluation values are pretty low!
9. test9.pt: 1024x1024x1024 pos_loss 1e-4, yaw_loss: 2e-4, tanh layer at the end, trained for hardware
10. test10.pt: 1024x1024x1024x1024 pos_loss: 3e-5, yaw_loss: 2e-5, LOG_STD_MAX = 20 (used to be 2), bbox [1.2, 1.2, 1.2]
11. test11.pt: 1024x1024x1024x1024x1024 pos_loss 6e-5, yaw_loss 4e-5, got rid of the squashed layer and replaced it with tanh(). many # of goal reached. only 50K data, and pretty converged
12. test12.pt: 2048x2048x2048x2048 pos_loss: 3e-5, 2e-5, got rid of the squashed layer. pretty converged after 50K data.
13. test13.pt: 1024x1024x1024x1024 pos_loss: 2e-5, 1.4e-5, dyn constraints for hw (v 2.0, a 3.0, j 4.0), obstacle observation is not noised
14. test14.pt: 1024x1024x1024x1024 pos_loss: 3e-5, 4e-5, dyn constraints for hw, using noised obs for obst, 1.0 margins, training_dt 0.1 (from 0.25), max_side_bbox_obs 3.0, prob_choose_cross (0.8 from 1.0)
15. test15.py: 1024x1024x1024x1024 pos_loss: 2e-5, 4e-5, same with test14 but with yaw_loss_weight 30.0 and evaluatoin_yaw_loss is around 0.05
16. test16.py: 1024x1024x1024x1024 pos_loss: 4e-5, 9e-5, training_dt 0.2. only ~50k round 
