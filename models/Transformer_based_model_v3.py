from torch import nn
import torch
import math


class TransformerEncoder_LSTM_3(nn.Module):
    def __positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        odd = False
        if d_model % 2 != 0:
            d_model += 1
            odd = True
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                                -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # drop the last dimension if d_model is odd
        if odd:
            pe = pe[:, :-1]

        return pe
    def __init__(self,feature_num,sequence_len,transformer_encoder_head_num,hidden_dim,lstm_num_layers,
                 lstm_dropout,fc_layer_dim,fc_dropout,device):
        super(TransformerEncoder_LSTM_3, self).__init__()

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.lstm_hidden_size = hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1
        self.lstm_dropout = lstm_dropout

        self.transformer_encoder_head_num = transformer_encoder_head_num

        # self.return_attention_weights = return_attention_weights

        # Sine positional encoding
        # self.positional_encoding = self.__positionalencoding1d(self.feature_num,self.sequence_len)
        # self.positional_encoding = self.positional_encoding.to(device, dtype=torch.float)
        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.sequence_len, nhead=self.transformer_encoder_head_num,
        )

         
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # lstm
        self.lstm = nn.LSTM(feature_num,
                            self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout)

        # fc layers
        self.linear = nn.Sequential(
            nn.Linear((self.lstm_hidden_size + self.sequence_len), self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
            nn.ReLU(),
        )



    # x represents our data
    def forward(self, x):
        # x1 = x + self.positional_encoding
        x1 = x.permute(0, 2, 1)
        x1 = self.transformer_encoder(x1)
        x1 = x1.permute(0, 2, 1)
        x1 = self.gap(x1)
        x1 = self.flatten(x1)

        # LSTM/
        x2, _ = self.lstm(x)

        # Raw
        x2 = x2.contiguous()
        x2 = x2[:, -1, :]

        x = torch.cat((x1, x2), dim=1)

        x = self.linear(x)
        
        return x