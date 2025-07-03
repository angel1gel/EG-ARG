from torch import nn
import torch
import torch.nn.functional as F



class E_GAT(nn.Module):
    """
    E(n) Graph Attention Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, normalize=False, tanh=False):
        super(E_GAT, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.normalize = normalize
        self.tanh = tanh
        self.epsilon = 1e-8

        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # Attention computation
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())

        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # Coordinate update
        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        edge_feat = self.edge_mlp(out)
        att_val = self.att_mlp(edge_feat)
        return edge_feat * att_val  # Apply attention weight

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        out = torch.cat([x, agg], dim=1)
        out = self.node_mlp(out)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat)

        return h, coord, edge_attr

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class EGAT(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, normalize=False, tanh=False):
        super(EGAT, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.dr = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        for i in range(0, n_layers):
            self.add_module("get_layer_%d" % i, E_GAT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        h = self.dr(h)
        h = self.leaky_relu(h)
        # for i in range(0, self.n_layers):
        #     h, x, _ = self._modules["get_layer_%d" % i](h, edges, x, edge_attr=edge_attr)
        #     h = self.dr(h)
        # h = self.dr(h)
        # h = self.relu(h)
        h = self.embedding_out(h)
        return h, x

class  EGATWithCNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', n_layers=4, act_fn=nn.SiLU(), residual=True, normalize=False, tanh=False):
        super(EGATWithCNN, self).__init__()
        
        self.device = device  # Store device info
        self.hidden_nf = hidden_nf
        self.in_node_nf = in_node_nf

        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)

        self.dr = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()
        self.n_layers = n_layers
        
        # self.egat = EGAT(in_node_nf=hidden_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf, in_edge_nf=in_edge_nf, device=device, n_layers=n_layers)

        for i in range(0, n_layers):
            self.add_module("get_layer_%d" % i, E_GAT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, normalize=normalize, tanh=tanh))
        
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
        #     nn.LeakyReLU(negative_slope=0.01),  # 负斜率为 0.01
        #     nn.MaxPool1d(kernel_size=3)
        # )
        # self.fc = nn.Linear(16* (((hidden_nf + in_node_nf) - 2)  // 3), 1)  # Adjust based on CNN output
        
        # 用深度可分离卷积代替普通卷积
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=3)
        )
        self.fc = nn.Linear(32* ((((hidden_nf + in_node_nf) - 2)  // 3 - 2) // 3), 1)  # Adjust based on CNN output

        
        # self.fc = nn.Linear(hidden_nf, 1)
        

        # # # 这里是lstm
        # self.lstm = nn.LSTM(input_size=self.hidden_nf, hidden_size=self.hidden_nf, num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)

        # # 这里是transformer
        # self.transformerencoderlayer = nn.TransformerEncoderLayer(d_model=self.hidden_nf, nhead=2)
        # self.transformer = nn.TransformerEncoder(self.transformerencoderlayer, num_layers=2)


    def forward(self, h, x, edges, edge_attr):
        # Move input tensors to the same device
        original_embedding = h

        h = self.embedding_in(h)
        h = self.dr(h)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        h = self.leaky_relu(h)
        # h = self.relu(h)
        # original_embedding = original_embedding.to(self.device)
        # h = h.to(self.device)
        # x = x.to(self.device)
        # edges = edges.to(self.device)
        # edge_attr = edge_attr.to(self.device)
        
        # Get EGAT embedding

        # # 把一下注释一下，for循环注释就好，不要就是cnn，加上LSTM就是LSTM
        # h, _ = self.lstm(h)

        # 把一下注释一下，for循环注释就好，不要就是cnn，加上transformer就是transformer
        # h = self.transformer(h)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["get_layer_%d" % i](h, edges, x, edge_attr=edge_attr)
            h = self.dr(h)

        h = self.dr(h)
        h = self.relu(h)
        # egat_embedding, _ = self.egat(h, x, edges, edge_attr)
        
        # Concatenate original embedding with EGAT embedding
        combined_embedding = torch.cat((original_embedding, h), dim=-1)
        
        # Prepare for CNN (reshape to [batch_size, channels, features])
        cnn_input = combined_embedding.unsqueeze(1).to(self.device)  # Add channel dimension for CNN
        
        # Pass through CNN
        cnn_output = self.cnn(cnn_input)
        
        # Flatten CNN output and pass through FC layer
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # Flatten

        cnn_output = self.fc(cnn_output)  # Pass through fully connected layer
        
        # return output, output
        return cnn_output, _

