python train.py headless=true algo=ppo_priv  total_frames=140000000 task.action_scaling=2.0
python train.py headless=true algo=ppo_priv  total_frames=140000000 task.action_scaling=2.0
python train.py headless=true algo=ppo  total_frames=140000000 task.action_scaling=2.0
python train.py headless=true algo=ppo  total_frames=140000000 task.action_scaling=2.0

# python train.py headless=true algo=ppo_gru  total_frames=140000000 
# python train.py headless=true algo=ppo_adapt  total_frames=140000000 
# python train.py headless=true algo=ppo_adapt algo.condition_mode=film  total_frames=140000000 
# python train.py headless=true algo=ppo_adapt algo.condition_mode=d2rl  total_frames=140000000