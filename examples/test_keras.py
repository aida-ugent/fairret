from folktables import ACSDataSource
from folktables import ACSIncome, generate_categories
import keras
import tensorflow as tf
import numpy as np


# data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# data = data_source.get_data(states=["AL"], download=True)

# definition_df = data_source.get_definitions(download=True)
# categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)

# df_feat, df_labels, _ = ACSIncome.df_to_pandas(data, categories=categories, dummies=True)
# df_feat.head()

# sens_cols = ['SEX_Female', 'SEX_Male']
# feat = df_feat.drop(columns=sens_cols).to_numpy(dtype="float")
# sens = df_feat[sens_cols].to_numpy(dtype="float")
# label = df_labels.to_numpy(dtype="float")

# np.save("feat", feat)
# np.save("sens", sens)
# np.save("label", label)

feat = np.load("feat.npy")
sens = np.load("sens.npy")
label = np.load("label.npy")

print(sens.mean(axis=0))


from fairret.statistic import TruePositiveRate
from fairret.loss import NormLoss

statistic = TruePositiveRate(torch=False)
norm_loss = NormLoss(statistic)

h_layer_dim = 16
lr = 1e-3
batch_size = 1024


keras.utils.set_random_seed(0)
print(f"Shape of the 'normal' features tensor: {feat.shape}")
print(f"Shape of the sensitive features tensor: {sens.shape}")
print(f"Shape of the labels tensor: {label.shape}")

h_layer_dim = 16
lr = 1e-3
batch_size = 1024

def combined_loss(y_true, y_pred):
    logit = y_pred
    batch_label = y_true[:, 0:1]
    batch_sens = y_true[:, 1:3]

    loss = tf.reduce_mean(tf.square(logit - batch_label)) + norm_loss(logit, batch_sens, batch_label)

    return loss

model = keras.Sequential(
    [
        keras.layers.Dense(h_layer_dim, activation="relu"),
        keras.layers.Dense(1),
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    run_eagerly=True
)

model.fit(x=feat, y=np.column_stack((label, sens)))
