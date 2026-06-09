import time
import math
import torch
import torch.nn.functional as F
from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, OverallLoss, FeatureConsistencyLoss
from model import create_model
from vision3d.engine import EpochBasedTrainer
from vision3d.utils.optimizer import build_optimizer, build_scheduler


class AdaptiveRewardMixer:
    def __init__(self, mode="std", tau=20.0):
        self.mode = mode
        self.tau = tau

    def compute_alpha(self, local, global_, epoch=None):
        if self.mode == "std":
            std_l, std_g = local.std(), global_.std()
            alpha = std_l / (std_l + std_g + 1e-6)
            if torch.isnan(local).any() or torch.isnan(global_).any():
                raise ValueError("local or global reward contains NaN.")

        elif self.mode == "schedule" and epoch is not None:
            t0 = 30  # 中心 epoch，可根据你的总 epoch 动态调整
            alpha = 1.0 / (1.0 + math.exp(-(epoch - t0) / self.tau))

        else:
            alpha = 0.5

        return alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    def update_tau(self, decay=0.9, tau_min=5.0):
        self.tau = max(tau_min, self.tau * decay)





import os
class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)

        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
        self.register_loader(train_loader, val_loader)
        self.log(f"Data loader created: {time.time() - start_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")

        model = create_model(cfg)
        self.register_model(model)

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_optimizer(optimizer)
        self.register_scheduler(scheduler)

        self.loss_func = OverallLoss(cfg)
        self.feat_func = FeatureConsistencyLoss(cfg)
        self.eval_func = EvalFunction(cfg)

        self.reward_mixer = AdaptiveRewardMixer(mode=cfg.rl.reward_mode, tau=cfg.rl.get("tau", 20.0))
        self.logged_rewards = []     

    def train_step(self, epoch, iteration, data_dict):
        use_rl = epoch >= 15 and epoch % 10 == 0  # RL 策略触发逻辑
        data_dict["rl_mode"] = use_rl
        data_dict["epoch"] = epoch

        output_dict = self.model(data_dict)

        # 主 loss
        loss_dict = self.loss_func(data_dict, output_dict, epoch)
        fc_loss = self.feat_func(output_dict, current_epoch=epoch)
        loss_dict["fc_loss"] = fc_loss
        loss_dict["loss"] += fc_loss

        if use_rl:
            try:
                reward = output_dict["query_rewards"]    # [Q]
                actions = output_dict["query_actions"]   # [Q]

                if torch.isnan(reward).any() or torch.isinf(reward).any():
                    raise ValueError(f"[RL] reward contains NaN or Inf: {reward}")

                main_loss = loss_dict["loss"].clamp(min=1e-2)
                dummy_loss = 1.0 / main_loss
                dummy_reward = torch.full_like(reward, dummy_loss.detach().item())  # reward 不参与 backward

                alpha = self.reward_mixer.compute_alpha(reward, dummy_reward, epoch)
                mixed_reward = alpha * reward + (1 - alpha) * dummy_reward

                self.model.update_query_scores_with_reinforce(mixed_reward, actions)


                # ✅ 打印和记录 reward 状态
                reward_mean = reward.detach().mean().item()
                reward_std = reward.detach().std().item()
                print(f"[RL] reward_mean = {reward_mean:.4f}, reward_std = {reward_std:.4f}, alpha = {alpha:.4f}")

                self.logged_rewards.append({
                    "epoch": epoch,
                    "iter": iteration,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "reward_min": reward.detach().min().item(),
                    "reward_max": reward.detach().max().item(),
                    "alpha": alpha,
                })

                # 最终 loss 用 dummy_loss 占位
                loss_dict["rl_reward"] = reward.detach().mean()
                loss_dict["dummy_loss"] = dummy_loss
                loss_dict["alpha"] = alpha
                loss_dict["loss"] = main_loss
                loss_dict["dummy_loss"] = 1.0 / main_loss.detach()

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[RL ERROR] {e}")

        return output_dict, loss_dict




    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict, epoch)
        fc_loss = self.feat_func(output_dict, current_epoch=epoch)
        loss_dict["fc_loss"] = fc_loss
        loss_dict["loss"] += fc_loss
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict
    
    def after_train_epoch(self, epoch, summary_dict):
        if self.logged_rewards:
            import json, os
            save_path = os.path.join(self.log_dir, f"reward_epoch_{epoch}.json")

            with open(save_path, "w") as f:
                json.dump(self.logged_rewards, f, indent=2)
            self.logged_rewards.clear()
        
        # 动态调整 tau（每10个 epoch）
        if epoch > 0 and epoch % 10 == 0:
            self.reward_mixer.update_tau(decay=0.9, tau_min=5.0)
            print(f"[Schedule] Updated tau = {self.reward_mixer.tau:.4f}")





def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
