import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np


"""
CARP3D L->D Depth Aggregation Linear Attention.
"""
class Depth_Attn_Linear(nn.Module):

    def __init__(self, L=1024, dropout=False, n_classes=1, activate=False):
        super(Depth_Attn_Linear, self).__init__()
        self.l_layer = nn.Linear(L, n_classes)
        self.module = [self.l_layer]

        if activate:
            self.module.append(nn.Tanh())

        if dropout:
            self.module.append(nn.Dropout(0.3))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x

    
"""
CARP3D L->D Depth Aggregation Non-gated Attention.
"""
class Depth_Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Depth_Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.7))#0.25

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x


"""
Attention Network with Sigmoid Gating for Lateral Patch aggregation.
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.7))#0.5
            self.attention_b.append(nn.Dropout(0.7))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
Attention-based 2D Multiple Instance Learning (2D ABMIL)
"""
class ABMIL(nn.Module):
    def __init__(self, dropout=False, n_classes=2, feat_dim=512):
        super(ABMIL, self).__init__()
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.7))

        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes) 
        self.n_classes = n_classes

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, label=None, return_features=False, attention_only=False):
        device = next(self.parameters()).device
        h = torch.stack(h).to(device)
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over A
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict


"""
CARP3D Naive
"""
class CARP3D_Naive(ABMIL):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super().__init__(self)
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.7))
        
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.n_classes = n_classes
        self.classifiers = nn.Linear(size[1], n_classes)
        self.relu = nn.ReLU()

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def process_seq(self, h_list):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        h_features = torch.cat(h_list, dim=0)
        h_features = h_features.to(device)
        A, h_features = self.attention_net(h_features)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h_features)
        return M, A

    def forward(self, h_list, return_features=False, attention_only=False):
        M, A = self.process_seq(h_list)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}

        if attention_only:
            return A
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A, results_dict
    

"""
CARP3D L->D with averaging depth aggregation
"""
class CARP3D_LD_Ave(ABMIL):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super().__init__(self)
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        
        self.hidden_size = size[1]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.7)) #0.5

        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.n_classes = n_classes
        self.classifiers = nn.Linear(size[1], n_classes)
        self.relu = nn.ReLU()

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def process_seq(self, h_list):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        A_list = []
        M_list = []
        ind = 0

        hidden_features = torch.zeros(1, self.hidden_size)
        hidden_features = hidden_features.to(device)
        while ind < len(h_list):
            h = h_list[ind]
            h = h.to(device)
            A, h = self.attention_net(h)  # NxK        
            A = torch.transpose(A, 1, 0)  # KxN

            A_list.append(A)

            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, h)

            M_list.append(M)
            hidden_features += M
            
            ind += 1

        return hidden_features, A_list, M_list

    def forward(self, h_list, return_features=False, attention_only=False):
        hidden_features, A_list, M_list = self.process_seq(h_list)
        context_features = hidden_features/len(h_list)
        logits = self.classifiers(context_features)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}
        M = torch.stack(M_list)

        if attention_only:
            return A_list
        if return_features:
            results_dict.update({'features': M})
            results_dict.update({'weighted_features': context_features})

        return logits, Y_prob, Y_hat, A_list, results_dict
    

"""
CARP3D L->D with linear attention depth aggregation
"""
class CARP3D_LD_Linear_Attn(ABMIL):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super().__init__(self)
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.7)) #0.5
        
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
       
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.seq_attention = Depth_Attn_Linear(L = size[1], n_classes = 1, activate=False, dropout=False)
        self.n_classes = n_classes
        self.classifiers = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.seq_attention = self.seq_attention.to(device)
        self.classifiers = self.classifiers.to(device)

    def process_seq(self, h_list):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        A_list = []
        M_list = []
        ind = 0

        hidden_features = []
        while ind < len(h_list):
            h = h_list[ind]
            h = h.to(device)
            A, h = self.attention_net(h)  # NxK   
            A = torch.transpose(A, 1, 0)  # KxN

            A_list.append(A)

            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, h)

            M_list.append(M)
            hidden_features.append(M)
            
            ind += 1
        hidden_features = torch.stack(hidden_features).squeeze()
        return hidden_features, A_list, M_list

    def forward(self, h_list, return_features=False, attention_only=False):
        hidden_features, A_list, M_list = self.process_seq(h_list)
        
        if attention_only:
            return A_list

        level_A_l, level_features = self.seq_attention(hidden_features)
        level_A_l = torch.transpose(level_A_l, 1, 0)
        level_A = F.softmax(level_A_l, dim=1)  # softmax over N

        context_features = torch.mm(level_A, level_features)
      
        logits = self.classifiers(context_features)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}
        M = torch.stack(M_list)

        if return_features:
            results_dict.update({'features': M})
            results_dict.update({'weighted_features': context_features})
            results_dict.update({'slice_attn': level_A})
        
        return logits, Y_prob, Y_hat, A_list, results_dict
    

"""
CARP3D L->D with non-gated attention depth aggregation
"""
class CARP3D_LD(ABMIL):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super().__init__(self)
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.7)) #0.5
        
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
       
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.seq_attention = Depth_Attn_Net(L = size[1], n_classes = 1, dropout=False)
        self.n_classes = n_classes
        self.classifiers = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.seq_attention = self.seq_attention.to(device)
        self.classifiers = self.classifiers.to(device)

    def process_seq(self, h_list):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        A_list = []
        M_list = []
        ind = 0

        hidden_features = []
        while ind < len(h_list):
            h = h_list[ind]
            h = h.to(device)
            A, h = self.attention_net(h)  # NxK   
            A = torch.transpose(A, 1, 0)  # KxN

            A_list.append(A)

            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, h)

            M_list.append(M)
            hidden_features.append(M)
            
            ind += 1
        hidden_features = torch.stack(hidden_features).squeeze()
        return hidden_features, A_list, M_list

    def forward(self, h_list, return_features=False, attention_only=False):
        hidden_features, A_list, M_list = self.process_seq(h_list)
        
        if attention_only:
            return A_list

        level_A_l, level_features = self.seq_attention(hidden_features)
        level_A_l = torch.transpose(level_A_l, 1, 0)
        level_A = F.softmax(level_A_l, dim=1)  # softmax over N

        context_features = torch.mm(level_A, level_features)
      
        logits = self.classifiers(context_features)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}
        M = torch.stack(M_list)

        if return_features:
            results_dict.update({'features': M})
            results_dict.update({'weighted_features': context_features})
            results_dict.update({'slice_attn': level_A})
        
        return logits, Y_prob, Y_hat, A_list, results_dict
    

class CARP3D_LD_RNN(ABMIL):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super().__init__(self)
        feat_dim = feat_dim
        size = [feat_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(0.7)) #0.5

        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        input_size = size[1]
        self.hidden_size = size[1]
        self.rnn_unit = nn.RNN(input_size, self.hidden_size, num_layers=1, dropout=0.5, batch_first=True, bias=False)

        self.n_classes = n_classes
        self.classifiers = nn.Linear(self.hidden_size*2, n_classes)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.rnn_unit = self.rnn_unit.to(device)

    def process_seq(self, h_list, reverse=False):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        A_list = []
        M_list = []
        hidden_list = []
        
        if not reverse:
            ind = 0
        else:
            ind = -1

        hidden_features = torch.tensor(np.random.normal(0, 1, self.hidden_size).reshape([1, self.hidden_size]), dtype=torch.float)
        hidden_features = hidden_features.unsqueeze(0)
        hidden_features = hidden_features.to(device)

        count = 0
        end_count = int((len(h_list) - 1)/2)
        
        while True:
            h = h_list[ind]
            h = h.to(device)
            A, h = self.attention_net(h)  # NxK        
            A = torch.transpose(A, 1, 0)  # KxN

            A_list.append(A)

            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, h)

            M_list.append(M)

            M = M.unsqueeze(0)
            hidden_features, _ = self.rnn_unit(M, hidden_features)
            
            hidden_list.append(hidden_features)

            if count == end_count:
                break
            
            if not reverse:
                ind += 1
            else:
                ind -= 1

            count += 1

        return hidden_features, A_list, M_list, hidden_list

    def forward(self, h_list, return_features=False, attention_only=False):
        front_hidden_features, front_A_list, front_M_list, front_hidden_list = self.process_seq(h_list, reverse=False)
        rear_hidden_features, rear_A_list, rear_M_list, rear_hidden_list = self.process_seq(h_list, reverse=True)
        
        combined_hidden_features = torch.cat((front_hidden_features[0], rear_hidden_features[0]), -1)
        logits = self.classifiers(combined_hidden_features) #relu
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}
        rear_A_list = rear_A_list[:-1]
        rear_A_list.reverse()
        A_list = front_A_list + rear_A_list

        rear_M_list = rear_M_list[:-1]
        rear_M_list.reverse()
        M_list = front_M_list + rear_M_list
        M = torch.stack(M_list)

        rear_hidden_list = rear_hidden_list[:-1]
        rear_hidden_list.reverse()
        hidden_list = front_hidden_list + rear_hidden_list
        H = torch.stack(hidden_list)

        if attention_only:
            return A_list
        if return_features:
            results_dict.update({'features': M})
            results_dict.update({'combined_features': combined_hidden_features})
            results_dict.update({'hidden_features': H})
            
        return logits, Y_prob, Y_hat, A_list, results_dict
    

"""
CARP3D D->L with non-gated attention depth aggregation
"""
class CARP3D_DL(nn.Module):
    def __init__(self, dropout=True, n_classes=2, feat_dim=512):
        super(CARP3D_DL, self).__init__()
        size = [feat_dim, 512, 256]
        
        # feature extractor: Linear + ReLU + Dropout
        self.feature_extractor = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(0.7) if dropout else nn.Identity()
        )
        
        # attention over lateral
        self.attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1
        )

        # attention over depth
        self.depth_attention = Depth_Attn_Net(
            L=size[1], D=size[2], dropout=False, n_classes=1
        )

        # final classifier
        self.classifiers = nn.Linear(size[1], n_classes)

        # optional weights init
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = self.feature_extractor.to(device)
        self.attention_net = self.attention_net.to(device)
        self.depth_attention = self.depth_attention.to(device)
        self.classifiers = self.classifiers.to(device)

    def process_seq(self, h_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_M_list = []
        depth_A_list = []

        for i in range(h_list[0].size(0)):
            # stack over depth dimension
            depth_features = torch.stack([h_list[d][i] for d in range(len(h_list))]).to(device)
            # attention over depth
            level_A_l, level_features = self.depth_attention(depth_features)
            level_A_l = torch.transpose(level_A_l, 1, 0)
            level_A = F.softmax(level_A_l, dim=1)
            context_feature = torch.mm(level_A, level_features)
            
            depth_M_list.append(context_feature)
            depth_A_list.append(level_A)
        
        depth_M = torch.stack(depth_M_list)  # shape [num_lateral, 1, feat_dim]
        return depth_M, depth_A_list

    def forward(self, h_list, return_features=False, attention_only=False):
        depth_M, depth_A_list = self.process_seq(h_list)
        depth_M = depth_M.squeeze(1)  # shape [num_lateral, feat_dim]

        # lateral feature extraction
        H = self.feature_extractor(depth_M)  # [N, L]
        A, h = self.attention_net(H)         # A: [N,1], h: [N,L]

        # softmax + aggregation
        A = A.T  # [1, N]
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)  # [1, L]

        # classifier
        logits = self.classifiers(M)  # [1, num_classes]
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({
                'features': depth_M,
                'weighted_features': M,
                'slice_attn': A,
                'depth_attn': depth_A_list
            })

        if attention_only:
            return A, depth_A_list

        return logits, Y_prob, Y_hat, (A, depth_A_list), results_dict
    