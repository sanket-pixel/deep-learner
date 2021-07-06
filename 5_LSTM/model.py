import torch
import torch.nn as nn
from lstm_cell import LSTMCell
class LSTM_Classifier(nn.Module):

    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, mode="zeros",device="cpu", model_mode = "custom"):

        super(LSTM_Classifier, self).__init__()
        self.mode = mode
        self.device = device
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.Linear(input_dim,emb_dim)
        self.model_mode = model_mode
        if model_mode == "pytorch":
            self.lstm = nn.LSTM(input_size = emb_dim, hidden_size=hidden_dim,
                                num_layers=num_layers)
        else:
            self.lstm = LSTM(input_size=emb_dim, hidden_size=hidden_dim,
                                num_layers=num_layers,bias=True)
        self.classifier = nn.Linear(hidden_dim,out_features=10)
        return

    def forward(self,x):

        b_size, n_channels, n_rows, n_cols = x.shape
        h,c = self.init_state(b_size)

        x_rowed = x.view(b_size, n_rows,n_channels*n_cols)
        embeddings = self.encoder(x_rowed)

        if self.model_mode == "pytorch":
            lstm_out, (h_out,c_out) = self.lstm(embeddings.permute(1,0,2),(h,c))
            y = self.classifier(lstm_out.permute(1,0,2)[:,-1,:])
        else:
            lstm_out = self.lstm(embeddings, (h, c))
            y = self.classifier(lstm_out)

        return y



    def init_state(self, b_size):
        if (self.mode == "zeros"):
            h = torch.zeros(self.num_layers, b_size, self.hidden_dim)
            c = torch.zeros(self.num_layers, b_size, self.hidden_dim)
        elif (self.mode == "random"):
            h = torch.randn(self.num_layers, b_size, self.hidden_dim)
            c = torch.randn(self.num_layers, b_size, self.hidden_dim)
        elif (self.mode == "learned"):
            h = self.learned_h.repeat(1, b_size, 1)
            c = self.learned_c.repeat(1, b_size, 1)
        h = h.to(self.device)
        c = c.to(self.device)
        return h, c



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        # self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

        # self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, sequence length , input_size)
        #
        # Output of shape (batch_size, output_size)

        h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[0][layer,:,:],h0[1][layer,:,:]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        # out = self.fc(out)

        return out



