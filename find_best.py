import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import copy
from pathlib import Path
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import random
import os
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

def load_single_data(file_path):
    # 读取数据
    df = pd.read_excel(file_path, parse_dates=["date"])
    # 检查时间连续性
    time_diff = df["date"].diff().dt.days.dropna()
    assert all(time_diff[1:] == 1), "时间序列存在间隔或重复"
    return df
# 执行数据加载
full_df = load_single_data("四姑娘山相关数据-终.xlsx")

def process_features(df):
    # 时间特征提取
    df['month'] = df['month'].astype(str).astype("category")            # 月份 (1-12)
    df['day_of_week'] = df['day_of_week'].astype(str).astype("category")  # 星期几 (0-6)
    df['is_weekend'] = df['is_weekend'].astype(str).astype("category")

    # 情感特征聚合(​情绪波动率​（捕捉趋势）)
    df['CommentSentiment'] = df['CommentSentiment'].astype(float)
    df["SearchIndex"] = df["SearchIndex"].astype(float)
    df["tourists"] = df["tourists"].astype(float)
    #细粒度情感
    df["topic_Strategy"] = df["topic_Strategy"].astype(float)
    df["topic_Hotel"] = df["topic_Hotel"].astype(float)
    df["topic_Transportation"] = df["topic_Transportation"].astype(float)
    df["topic_Serve"] = df["topic_Serve"].astype(float)
    df["topic_Scenery"] = df["topic_Scenery"].astype(float)
    df["topic_Experience"] = df["topic_Experience"].astype(float)
    df["topic_Play"] = df["topic_Play"].astype(float)
    #将节假日和天气转变成 分类数据
    df['Vacation'] = df['Vacation'].astype(str).astype("category")
    df['WeatherLevel'] = df['WeatherLevel'].astype(str).astype("category")
    # 添加序列ID
    df['series_id'] = 0  # 单序列预测
    df["series_id"] = df["series_id"].astype(str).astype("category")
    df['new_time_idx'] = np.arange(669)
    return df

# 执行特征工程
processed_df = process_features(full_df)


step_range = [i for i in range(2,14)]
batch_range = [i for i in range(16,64)]
for step_t in step_range:
    for batch_t in batch_range:
        folder_path = "./trail/"+"step"+str(step_t)+"-"+"batch"+str(batch_t)+"/"
        # 设置模型参数
        #这里用7天预测1天的
        max_prediction_length = 1   # 预测未来1天
        max_encoder_length = step_t     # 使用过去7天作为输入
        train_data = processed_df[processed_df.new_time_idx<=536]
        val_data = processed_df[processed_df.new_time_idx>=536]

        # 创建时间序列数据集（训练集）
        training = TimeSeriesDataSet(
            train_data,
            time_idx="new_time_idx",          # 时间索引列名
            target="tourists",              # 预测目标列
            group_ids=["series_id"],  # 分组标识，只有一组
            min_encoder_length=max_encoder_length // 2,  # 最小编码长度
            max_encoder_length=max_encoder_length,       # 最大编码长度
            min_prediction_length=1,      # 最小预测长度
            max_prediction_length=max_prediction_length, # 最大预测长度
            static_categoricals=["series_id"],       # 静态分类特征
            static_reals=[],  # 静态数值特征，在我的模型中，没有这一类别的数值，所以直接为空

            time_varying_known_categoricals=["day_of_week", 
                                            "month",
                                            "is_weekend",
                                            #  "special_days",
                                            "Vacation",
                                            "WeatherLevel"
                                            ],  # 时间已知分类特征
            # variable_groups={"special_days": special_days},  # 变量组（特殊日期）
            time_varying_known_reals=["new_time_idx"],  # 时间已知数值特征
            time_varying_unknown_categoricals=[],       # 时间未知分类特征,包括天气等级
            time_varying_unknown_reals=[                # 时间未知数值特征（包含目标）
                "tourists", "SearchIndex", 
                'PictureAesthetics',
                # "CommentSentiment",
                "topic_Strategy","topic_Hotel","topic_Transportation","topic_Serve","topic_Scenery","topic_Experience","topic_Play"
            ],
            target_normalizer=GroupNormalizer(           # 目标归一化（按组）
                groups=["series_id"], transformation="softplus"
            ),
            # scalers=scalers,
            
            add_relative_time_idx=True,    # 添加相对时间索引
            add_target_scales=True,        # 添加目标缩放信息
            add_encoder_length=True,       # 添加编码器长度
        )

        # 创建时间序列数据集（验证集）
        validation = TimeSeriesDataSet(
            # processed_df[lambda x: x.new_time_idx <= training_cutoff],  # 筛选训练数据
            val_data,
            time_idx="new_time_idx",          # 时间索引列名
            target="tourists",              # 预测目标列
            group_ids=["series_id"],  # 分组标识，只有一组
            min_encoder_length=max_encoder_length // 2,  # 最小编码长度
            max_encoder_length=max_encoder_length,       # 最大编码长度
            min_prediction_length=1,      # 最小预测长度
            max_prediction_length=max_prediction_length, # 最大预测长度
            static_categoricals=["series_id"],       # 静态分类特征
            static_reals=[],  # 静态数值特征，在我的模型中，没有这一类别的数值，所以直接为空
            time_varying_known_categoricals=["day_of_week",
                                            "month",
                                            "is_weekend",
                                            #  "special_days",
                                            "Vacation",
                                            "WeatherLevel"
                                            ],  # 时间已知分类特征
            # variable_groups={"special_days": special_days},  # 变量组（特殊日期）
            time_varying_known_reals=["new_time_idx"],  # 时间已知数值特征
            time_varying_unknown_categoricals=[],       # 时间未知分类特征,包括天气等级
            time_varying_unknown_reals=[                # 时间未知数值特征（包含目标）
                "tourists", "SearchIndex",  
                'PictureAesthetics',
                #  "CommentSentiment",
                "topic_Strategy","topic_Hotel","topic_Transportation","topic_Serve","topic_Scenery","topic_Experience","topic_Play"
            ],
            target_normalizer=GroupNormalizer(           # 目标归一化（按组）
                groups=["series_id"], transformation="softplus"
            ),
            # scalers=scalers,
            add_relative_time_idx=True,    # 添加相对时间索引
            add_target_scales=True,        # 添加目标缩放信息
            add_encoder_length=True,       # 添加编码器长度
        )


        # 创建数据加载器
        batch_size = batch_t  # 批量大小
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0,shuffle=False 
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size , num_workers=0,shuffle=False  # 验证集用更大批量
        )
        # create study
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path=folder_path,
            n_trials=1000,
            max_epochs=400,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(2, 128),
            attention_head_size_range=(1, 20),
            learning_rate_range=(0.001, 1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(
                limit_train_batches=batch_t,
                accelerator="gpu",
                devices=1,
                        ),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
        )
        # save study results - also we can resume tuning at a later point in time
        with open("test_study_"+str(step_t)+"_"+str(batch_t)+".pkl", "wb") as fout:
            pickle.dump(study, fout)
        # show best hyperparameters
        print(study.best_trial.params)
print("done!")