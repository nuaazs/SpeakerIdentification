#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)
    训练x-vector还是ECAPA-TDNN取决于所用yaml文件

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# Tensorboard依赖
from torch.utils.tensorboard import SummaryWriter 

class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            # 数据增强
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline): # 枚举各种aug

                # Apply augment
                # 调用augment()
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]: # 超长了-截取
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else: # 短了-补0
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs
            
            # 将原数据和aug数据进行cat
            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        # 特征提取+norm
        feats = self.modules.compute_features(wavs) # 提特征
        feats = self.modules.mean_var_norm(feats, lens) # 归一化

        # Embeddings + speaker classifier
        # 提emb+分类
        embeddings = self.modules.embedding_model(feats) # 提embedding
        outputs = self.modules.classifier(embeddings) # 分类

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        # 对aug数据的label复制粘贴
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid] * self.n_augment, dim=0)

        # 预测标签与gt标签之间的negative log likelihood
        loss = self.hparams.compute_cost(predictions, spkid, lens)

        # lr退火
        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

            # Tensorboard参数存储路径
            tb_writer = SummaryWriter(os.path.join(self.hparams.output_folder,"tensorboard_logs"))
            # 保存tensorboard文件
            #tb_writer.add_scalar('train/loss', self.train_loss, epoch) # 写入train_loss
            tb_writer.add_scalar('valid/error', stage_loss, epoch) # 写入val_loss
            tb_writer.add_scalar('params/lr', old_lr, epoch) # 写入val_loss


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    # 训练集
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    # 验证集
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"]) # 采样点数

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]: # 随机切取
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1) # 随机起点
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        ) # 读取音频片段
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    # 将VoxCeleb的spk_id标签映射为数字 /data/results/VoxCeleb1/xvect_aug/exp/save/label_encoder.txt
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data,valid_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    # 训练集 测试集 标签映射
    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    # 解析命令行
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    # 读取关于超参数和配置的yaml文件
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # 选择GPU
    if hparams['GPU'] == 0:
        run_opts["device"] = 'cuda:0'
    elif hparams['GPU'] == 1:
        run_opts["device"] = 'cuda:1'

    # run_opts["distributed_launch"] = True
    # run_opts["local_rank"] = 1
    # run_opts["device"] = 'cuda'

    # Initialize ddp (useful only for multi-GPU DDP training)
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Download verification list (to exlude verification sentences from train)
    # 测试集列表 /data/results/VoxCeleb1/xvect_aug/config/save/veri_test2.txt
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    # 下载验证集列表
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    # 准备VoxCeleb数据集并生成相应csv文件
    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"], # 原始VoxCeleb数据所在地址
            "save_folder": hparams["save_folder"], # 将要保存csv的地址
            "verification_pairs_file": veri_file_path, # 测试集列表
            "splits": ["train", "dev"], # 将训练集划分为train/dev分别用于训练和验证
            "split_ratio": [90, 10], # train/dev划分比例
            "seg_dur": hparams["sentence_len"], # 音频剪切长度/秒
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    # 训练集 验证集 标签映射
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    # 按超参数实例化说话人识别模型class SpeakerBrain
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    print(run_opts)

    # from IPython import embed
    # embed()

    # Training
    # 启动训练
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
