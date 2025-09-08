import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}#, "swish": swish
NORM2FN = {'BN1d':nn.BatchNorm1d, 'BN2d':nn.BatchNorm2d, 'LN':nn.LayerNorm}


class SketchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SketchEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_states):
        return self.embedding(input_states)


class SketchDiscreteEmbedding(nn.Module):
    '''
    max_size[tuple](x_length, y_length)
    '''
    def __init__(self, max_size, type_size, hidden_dim, pool_type):
        super(SketchDiscreteEmbedding, self).__init__()
        self.x_embedding = nn.Embedding(2*max_size[0]+2, hidden_dim//2)
        self.y_embedding = nn.Embedding(2*max_size[1]+2, hidden_dim//2)
        self.type_embedding = nn.Embedding(type_size+1, hidden_dim)
        assert pool_type in ['sum', 'con']
        self.pool_type = pool_type

    '''
    input_states[batch, seq_len, 3(input_dim)](Inputs are encoded as discrete type)
    '''
    def forward(self, input_states):
        input_states = input_states.to(dtype=torch.long)
        input_states = input_states + 1
        x_hidden = self.x_embedding(input_states[:,:,0])
        y_hidden = self.y_embedding(input_states[:,:,1])
        axis_hidden = torch.cat([x_hidden, y_hidden], dim=2)

        type_hidden = self.type_embedding(input_states[:,:,2])

        if self.pool_type == 'sum':
            return axis_hidden + type_hidden
        elif self.pool_type == 'con':
            return torch.cat([axis_hidden, type_hidden], dim=2)


class SketchSinPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchSinPositionEmbedding, self).__init__()
        self.pos_embedding_matrix = torch.zeros(max_length, pos_hidden_dim)
        pos_vector = torch.arange(max_length).view(max_length, 1).type(torch.float)
        dim_vector = torch.arange(pos_hidden_dim).type(torch.float) + 1.0
        self.pos_embedding_matrix[:,::2] = torch.sin(pos_vector / (dim_vector[::2] / 2).view(1, -1))
        self.pos_embedding_matrix[:,1::2] = torch.cos(pos_vector / ((dim_vector[1::2] - 1) / 2).view(1, -1))
    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding_matrix[position_labels.view(-1),:].view(position_labels.size(0), position_labels.size(1), -1)


class SketchLearnPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchLearnPositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_length, pos_hidden_dim)

    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding(position_labels)


class SketchEmbeddingRefineNetwork(nn.Module):
    '''
    The module to upsample the embedding feature, idea from the ALBERT: Factorized Embedding
    '''
    def __init__(self, out_dim, layers_dim):
        super(SketchEmbeddingRefineNetwork, self).__init__()
        self.layers = []
        layers_dim = layers_dim.copy()
        layers_dim.append(out_dim)

        for i in range(len(layers_dim)-1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_state):
        x = input_state
        for layer in self.layers:
            x = layer(x)
        return x


def setting2dict(paras, setting):
    paras['num_heads'] = setting[0]
    paras['hidden_dim'] = setting[1]
    paras['inter_dim'] = setting[2]


class SketchSelfAttention(nn.Module):
    '''
    Implementation for self attention in Sketch.
    The input will be a K-Dim feature.
    Input Parameters:
        config[dict]:
            hidden_dim[int]: The dimension of input hidden embeddings in the self attention, hidden diension is equal to the output dimension
            num_heads[int]: The number of heads
            attention_probs[float]: probability parameter for dropout
    '''
    def __init__(self, num_heads, hidden_dim, attention_dropout_prob):
        super(SketchSelfAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        #self.attention_dropout_prob = config.attention_dropout_prob
        # Calculation for intermeidate parameters
        self.head_dim = int(self.hidden_dim / self.num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale_factor = math.sqrt(self.head_dim)

        self.query = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.key = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.value = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.dropout = nn.Dropout(attention_dropout_prob)
        self.multihead_output = None

    def transpose_(self, x):
        '''
        Transpose Function for simplicity.
        '''
        new_x_shape = x.size()[:-1] + (self.num_heads , self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False, keep_multihead_output=False):
        '''
        Input:
            hidden_states[batch, seq_len, hidden_dim]
            attention_mask[batch,  1, 1, seq_len]
        Output:
            context_states[batch, seq_len, hidden_dim]
            attention_probs[seq_len, hidden_dim]
        '''
        # Get query, key, value together
        query = self.query(hidden_states) # [batch, seq_len, all_head_dim]
        key = self.key(hidden_states) # [batch, seq_len, all_head_dim]
        value = self.value(hidden_states) # [batch, seq_len, all_head_dim]

        # tranpose the query, key, value into multi heads[batch, seq_len, ]
        multi_query = self.transpose_(query) # [batch, num_heads, seq_len, head_dim]
        multi_key = self.transpose_(key) # [batch,  num_heads, seq_len, head_dim]
        multi_value = self.transpose_(value) # [batch, num_heads, seq_len, head_dim]

        # Calculate Attention maps
        attention_scores = torch.matmul(multi_query, multi_key.transpose(-1, -2))
        attention_scores = attention_scores / self.scale_factor
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # Compute states values
        context_states = torch.matmul(attention_probs, multi_value)

        if keep_multihead_output:
            self.multihead_output = context_states
            self.multihead_output.retain_grad()

        context_states = context_states.permute(0,2,1,3)
        context_states = context_states.contiguous().view(context_states.size()[:-2]+(-1,)) #view(context_states.size()[:-2]+ (self.all_head_dim,))

        if output_attentions:
            return context_states, attention_probs
        return context_states


class SketchMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim,
                 attention_norm_type, attention_dropout_prob, hidden_dropout_prob,):
        super(SketchMultiHeadAttention, self).__init__()
        self.attention = SketchSelfAttention(num_heads, hidden_dim, attention_dropout_prob)
        self.output = SketchOutput(hidden_dim, hidden_dim, attention_norm_type, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):
        input_states = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask, head_mask=head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        output_states = self.output(hidden_states, input_states)
        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchIntermediate(nn.Module):
    def __init__(self, hidden_dim, inter_dim, inter_activation):
        super(SketchIntermediate, self).__init__()
        self.fc = nn.Linear(hidden_dim, inter_dim)
        self.activation = ACT2FN[inter_activation]


    def forward(self, hidden_states):

        hidden_states = hidden_states.to(next(self.fc.parameters()).device)

        inter_states = self.fc(hidden_states.contiguous())
        inter_states = self.activation(inter_states)
        return inter_states


class SketchOutput(nn.Module):
    def __init__(self, input_dim, output_dim, attention_norm_type, output_dropout_prob):
        super(SketchOutput, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if attention_norm_type not in NORM2FN:
            raise ValueError(
                "The attention normalization is not in standard normalization types.")
        self.norm = NORM2FN[attention_norm_type](output_dim)
        self.dropout = nn.Dropout(output_dropout_prob)
    '''
    Input:
        hidden_states[]:

    Output:
        hidden_states[]:
    '''
    def forward(self, hidden_states, input_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states+input_states)
        return hidden_states


class SketchLayer(nn.Module):
    '''
        A transformer layer for sketch bert
    '''
    def __init__(self, num_heads, hidden_dim, inter_dim,
                 attention_norm_type, inter_activation, attention_dropout_prob,
                 hidden_dropout_prob, output_dropout_prob,):
        super(SketchLayer, self).__init__()
        self.attention = SketchMultiHeadAttention(num_heads, hidden_dim,
                                                    attention_norm_type, attention_dropout_prob, hidden_dropout_prob,)
        self.inter_layer = SketchIntermediate(hidden_dim, inter_dim, inter_activation)
        self.output = SketchOutput(inter_dim, hidden_dim, attention_norm_type, output_dropout_prob)


    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]:
        attention_mask[batch, seq_len]


    '''
    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):

        hidden_states = self.attention(hidden_states, attention_mask, head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        inter_states = self.inter_layer(hidden_states)
        output_states = self.output(inter_states, hidden_states)

        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchALEncoder(nn.Module):
    '''
        A Lite BERT: Parameter Sharing
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchALEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        setting2dict(layer_paras, layers_setting[0])
        self.sketch_layer = SketchLayer(**layer_paras)
        self.layers = []
        for layer_setting in layers_setting:
            self.layers.append(self.sketch_layer)
        #self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchEncoder(nn.Module):
    '''
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        self.layers = []
        for layer_setting in layers_setting:
            setting2dict(layer_paras, layer_setting)
            self.layers.append(SketchLayer(**layer_paras))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchTransformerModel(nn.Module):
    '''
    Input:
        layers_setting[list]
        input_dim[int]
        max_length[int]
        position_type[str]
        attention_norm_type[str]
        inter_activation[str]
        attention_dropout_prob[float]
        hidden_dropout_prob[float]
        output_dropout_prob[float]
    '''
    def __init__(self,
                 model_type='albert',
                 layers_setting=[[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072]],
                 embed_layers_setting=[128,256,512],
                 input_dim=5,
                 max_length=250 + 0 + 0,
                 max_size=[128, 128],
                 type_size=3,
                 position_type='learn',
                 segment_type='none',
                 sketch_embed_type='linear',
                 embed_pool_type='sum',
                 attention_norm_type='LN',
                 inter_activation='gelu',
                 attention_dropout_prob=0.5,
                 hidden_dropout_prob=0.5,
                 output_dropout_prob=0.5
                 ):
        print('create sketch_transformer model')
        super().__init__()

        self.layers_setting = layers_setting
        self.num_hidden_layers = len(layers_setting)
        self.embed_pool_type = embed_pool_type
        assert sketch_embed_type in ['linear', 'discrete']

        if sketch_embed_type == 'linear':
            self.embedding = SketchEmbedding(input_dim, embed_layers_setting[0])
        elif sketch_embed_type == 'discrete':
            self.embedding = SketchDiscreteEmbedding(max_size, type_size, embed_layers_setting[0], embed_pool_type)
        assert position_type in ['sin', 'learn', 'none']

        if position_type == 'sin':
            self.pos_embedding = SketchSinPositionEmbedding(max_length, embed_layers_setting[0])
        elif position_type == 'learn':
            self.pos_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.pos_embedding = None
        if segment_type == 'learn':
            self.segment_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.segment_embedding = None

        self.embed_refine_net = SketchEmbeddingRefineNetwork(layers_setting[0][1], embed_layers_setting)

        assert model_type in ['albert', 'bert']
        if model_type == 'albert':
            self.encoder = SketchALEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)
        elif model_type == 'bert':
            self.encoder = SketchEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)

    def load_model(self, state_dict, own_rel_in_input, own_cls_in_input, pre_rel_in_input, pre_cls_in_input):
        own_state = self.state_dict()
        for k, v in own_state.items():
            if k == 'pos_embedding.pos_embedding.weight':
                own_pos_size = v.size(0)
                seq_len = own_pos_size - own_rel_in_input - own_cls_in_input
                pretrained_pos_size = state_dict[k].size(0)
                own_start_ind = int(own_rel_in_input+own_cls_in_input)
                pre_start_ind = int(pre_rel_in_input+pre_cls_in_input)
                seq_len = min(seq_len, state_dict[k].size(0)-pre_start_ind)
                own_state[k][own_start_ind:own_start_ind+seq_len] = state_dict[k][pre_start_ind:pre_start_ind+seq_len]
                if own_rel_in_input and own_cls_in_input:
                    if pre_cls_in_input and pre_cls_in_input:
                        own_state[k][:2] = state_dict[k][:2]
                    elif pre_cls_in_input:
                        own_state[k][1] = state_dict[k][0]
                    elif pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_rel_in_input:
                    if pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_cls_in_input:
                    if pre_cls_in_input:
                        own_state[k][0] = state_dict[k][int(pre_rel_in_input)]
            else:
                own_state[k] = state_dict[k]
        self.load_state_dict(own_state)

    def get_pos_states(self, input_states):
        return torch.arange(input_states.size(1)).view(1,-1).repeat(input_states.size(0),1).to(device=input_states.device)
    '''
    Input:
        input_states[batch, seq_len, 5],
        attention_mask[batch, seq_len]/[batch, seq_len, ],(length mask)
    Output:
        output_states[batch, seq_len, hidden_dim],
    '''
    def forward(self, input_states, attention_mask, segments=None, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))
        # Extending attention mask
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask
        # process head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # switch to fload if need + fp16 compatibility
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states.to(device=input_states.device)

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states
        input_states = self.embed_refine_net(input_states)
        output_states = self.encoder(input_states, attention_mask, head_mask, output_all_states, output_attentions, keep_multihead_output)

        if output_attentions:
            output_states, attention_probs = output_states
            return output_states[-1], attention_probs

        return output_states[-1]


class SketchTransformer(nn.Module):
    def __init__(self, max_length=40, dropout=0.4):  # max_length = 1200
        super().__init__()

        self.sketch_transformer = SketchTransformerModel(max_length=max_length)
        self.linear = nn.Sequential(
            nn.Linear(768, 627),
            nn.BatchNorm1d(627),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(627, 512)
        )

    def forward(self, input_states, attention_mask):
        """
        input_states: [bs, max_len, 5]
        attention_mask: [bs, max_len]
        """
        fea = self.sketch_transformer(input_states, attention_mask)
        fea = fea.max(1)[0]
        fea = self.linear(fea)

        return fea


class SketchTransformerCls(nn.Module):
    def __init__(self, n_classes, max_length=200, dropout=0.4):
        super().__init__()

        self.sketch_transformer = SketchTransformerModel(max_length=max_length)

        channel_l0 = 768
        channel_l1 = int((channel_l0 * n_classes) ** 0.5)
        channel_l2 = n_classes

        self.linear = nn.Sequential(
            nn.Linear(channel_l0, channel_l1),
            nn.BatchNorm1d(channel_l1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(channel_l1, channel_l2),
        )

    def forward(self, input_states, attention_mask):
        fea = self.sketch_transformer(input_states, attention_mask)
        fea = fea.max(1)[0]
        fea = self.linear(fea)
        fea = F.log_softmax(fea, 1)

        return fea


if __name__ == '__main__':
    atensor = torch.rand(10, 256, 5)
    amask = torch.rand(10, 256)

    anet = SketchTransformer(256)

    aout = anet(atensor, amask)

    print(aout.size())


    pass






