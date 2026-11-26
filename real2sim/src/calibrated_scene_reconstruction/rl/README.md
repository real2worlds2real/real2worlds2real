# Reinforcement learning interface

This directory reserves the RL domain; it contains no trainer or backend dependency.

Future trainers consume task environments composed by apps. Tasks own reset/step, observations, actions and success; sim owns backend mechanics. Policy checkpoints and evaluation use the same task adapters. A trainer must not import scene reconstruction strategies or read development ground truth as an online success rule.
