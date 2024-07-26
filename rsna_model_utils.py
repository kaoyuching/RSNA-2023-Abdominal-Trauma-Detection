import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from monai.networks.nets import resnet


class Convert2DTo3DMixin:
    def _get_single_size(self, param):
        return param[0] if hasattr(param, '__iter__') else param
    
    def _2d_to_3d(self, module: nn.Module):
        _module = module
        for k, m in list(module.named_children()):
            mcls = m.__class__
            if mcls is nn.Conv2d:
                ks = self._get_single_size(m.kernel_size)
                _m = nn.Conv3d(
                    m.in_channels,
                    m.out_channels,
                    kernel_size=ks,
                    stride=self._get_single_size(m.stride),
                    padding=self._get_single_size(m.padding),
                    dilation=self._get_single_size(m.dilation),
                    groups=m.groups,
                    bias=False if m.bias is None else True,
                    padding_mode=m.padding_mode,
                )
                _m.register_parameter('weight', nn.Parameter(m.weight.unsqueeze(dim=4).repeat(1,1,1,1,ks)))
                _m.register_parameter('bias', None if m.bias is None else nn.Parameter(m.bias))
                
                
            elif mcls is nn.BatchNorm2d:
                _m = nn.BatchNorm3d(
                    m.num_features,
                    eps=m.eps,
                    momentum=m.momentum,
                    affine=m.affine,
                    track_running_stats=m.track_running_stats,
                )
                _m.register_buffer('running_mean', m.running_mean)
                _m.register_buffer('running_var', m.running_var)
                
            elif mcls is nn.MaxPool2d:
                _m = nn.MaxPool3d(
                    kernel_size=self._get_single_size(m.kernel_size),
                    stride=self._get_single_size(m.stride),
                    padding=self._get_single_size(m.padding),
                    dilation=self._get_single_size(m.dilation),
                    return_indices=m.return_indices,
                    ceil_mode=m.ceil_mode,
                )
            elif mcls is nn.AdaptiveAvgPool2d:
                _m = nn.AdaptiveAvgPool3d(
                    output_size=self._get_single_size(m.output_size)
                )
            else:
                if '2d' in mcls.__name__:
                    warnings.warn(f'Unhandled 2d layer found {mcls}. If not, please ignore this message.')
                _m = self._2d_to_3d(m)
            module.add_module(k, _m)
        return module    


# classification model
class RSNAModel(nn.Module):
    def __init__(self, model_name, pretrained=None, spatial_dims=3, n_input_channels=3, infer=False):
        super(RSNAModel, self).__init__()
        init_model = getattr(resnet, model_name)
        self.model = init_model(
            pretrained=False,
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
        )
        
        if pretrained is not None:
            if os.path.exists(pretrained):
                pretrain = torch.load(pretrained)
                net_dict = self.model.state_dict()
                pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

                net_dict.update(pretrain_dict)
                self.model.load_state_dict(net_dict)
                
        # model head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # five type injury
        self.bowel_fc = nn.Linear(in_features, out_features=1, bias=True)
        self.ext_fc = nn.Linear(in_features, out_features=1, bias=True)
        self.kidney_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.liver_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.spleen_fc = nn.Linear(in_features, out_features=3, bias=True)
        
        # if infer, return probability
        self.infer = infer
    
    def forward(self, x):
        x_fea = self.model(x)
        
        # classification
        out_bowel = self.bowel_fc(x_fea)
        out_ext = self.ext_fc(x_fea)
        out_kidney = self.kidney_fc(x_fea)
        out_liver = self.liver_fc(x_fea)
        out_spleen = self.spleen_fc(x_fea)
        
        if self.infer:
            out_bowel = torch.sigmoid(out_bowel)
            out_ext = torch.sigmoid(out_ext)
            out_kidney = torch.softmax(out_kidney, dim=1)
            out_liver = torch.softmax(out_liver, dim=1)
            out_spleen = torch.softmax(out_spleen, dim=1)
        
        return out_bowel, out_ext, out_kidney, out_liver, out_spleen
    
    
class RSNAModelTimm(Convert2DTo3DMixin, nn.Module):
    def __init__(self, model_name, pretrained=None, n_input_channels=3, infer=False):
        super(RSNAModelTimm, self).__init__()
        import timm
        
        self.model = self._2d_to_3d(timm.create_model(
            model_name,
            pretrained=True if pretrained is None else False,
            in_chans=n_input_channels
        ))
        
        if pretrained is not None:
            if os.path.exists(pretrained):
                pretrain = torch.load(pretrained, map_location='cpu')
                pretrain = {'model.' + k[7:]: v for k, v in pretrain['state_dict'].items()}
                net_dict = self.model.state_dict()
                pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}

                for k, param in pretrain_dict.items():
                    net_param = net_dict[k]  # (b, c, d, h, w)
                    if net_param.shape != param.shape:
                        net_ch = net_param.shape[1]
                        _param = torch.mean(param, dim=1, keepdim=True)
                        _param = _param.repeat(1, net_ch, 1, 1, 1)
                        pretrain_dict[k] = _param

                net_dict.update(pretrain_dict)
                self.model.load_state_dict(net_dict, strict=True)
                print('Successfully load pretrained weight')
                
        # model head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # five type injury
        self.bowel_fc = nn.Linear(in_features, out_features=1, bias=True)
        self.ext_fc = nn.Linear(in_features, out_features=1, bias=True)
        self.kidney_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.liver_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.spleen_fc = nn.Linear(in_features, out_features=3, bias=True)
        
        # if infer, return probability
        self.infer = infer
    
    def forward(self, x):
        x_fea = self.model(x)
        
        # classification
        out_bowel = self.bowel_fc(x_fea)
        out_ext = self.ext_fc(x_fea)
        out_kidney = self.kidney_fc(x_fea)
        out_liver = self.liver_fc(x_fea)
        out_spleen = self.spleen_fc(x_fea)
        
        if self.infer:
            out_bowel = torch.sigmoid(out_bowel)
            out_ext = torch.sigmoid(out_ext)
            out_kidney = torch.softmax(out_kidney, dim=1)
            out_liver = torch.softmax(out_liver, dim=1)
            out_spleen = torch.softmax(out_spleen, dim=1)
        
        return out_bowel, out_ext, out_kidney, out_liver, out_spleen
    
    
class RSNAModelTimmOrgan(Convert2DTo3DMixin, nn.Module):
    def __init__(self, model_name, pretrained=None, n_input_channels=3, infer=False):
        super(RSNAModelTimmOrgan, self).__init__()
        import timm
        
        self.model = self._2d_to_3d(timm.create_model(
            model_name,
            pretrained=True if pretrained is None else False,
            in_chans=n_input_channels
        ))
        
        if pretrained is not None:
            if os.path.exists(pretrained):
                pretrain = torch.load(pretrained, map_location='cpu')
                pretrain = {'model.' + k[7:]: v for k, v in pretrain['state_dict'].items()}
                net_dict = self.model.state_dict()
                pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}

                for k, param in pretrain_dict.items():
                    net_param = net_dict[k]  # (b, c, d, h, w)
                    if net_param.shape != param.shape:
                        net_ch = net_param.shape[1]
                        _param = torch.mean(param, dim=1, keepdim=True)
                        _param = _param.repeat(1, net_ch, 1, 1, 1)
                        pretrain_dict[k] = _param

                net_dict.update(pretrain_dict)
                self.model.load_state_dict(net_dict, strict=True)
                print('Successfully load pretrained weight')
                
        # model head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # five type injury
        self.kidney_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.liver_fc = nn.Linear(in_features, out_features=3, bias=True)
        self.spleen_fc = nn.Linear(in_features, out_features=3, bias=True)
        
        # if infer, return probability
        self.infer = infer
    
    def forward(self, x):
        x_fea = self.model(x)
        
        # classification
        out_kidney = self.kidney_fc(x_fea)
        out_liver = self.liver_fc(x_fea)
        out_spleen = self.spleen_fc(x_fea)
        
        if self.infer:
            out_kidney = torch.softmax(out_kidney, dim=1)
            out_liver = torch.softmax(out_liver, dim=1)
            out_spleen = torch.softmax(out_spleen, dim=1)
        
        return out_kidney, out_liver, out_spleen
    
    
# attention version with concat features
class RSNAModel2D(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=None,
        spatial_dims=2,
        n_input_channels=3,
        num_heads: int = 2,
        embed_size: int = 512,
        dropout_rate: float = 0.1,
        infer=False,
    ):
        super(RSNAModel2D, self).__init__()
        import timm
        
        self.model = self._2d_to_3d(timm.create_model(
            model_name,
            pretrained=True if pretrained is None else False,
            in_chans=n_input_channels
        ))
        
        if pretrained is not None:
            if os.path.exists(pretrained):
                pretrain = torch.load(pretrained, map_location='cpu')
                pretrain = {'model.' + k[7:]: v for k, v in pretrain['state_dict'].items()}
                net_dict = self.model.state_dict()
                pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}

                for k, param in pretrain_dict.items():
                    net_param = net_dict[k]  # (b, c, d, h, w)
                    if net_param.shape != param.shape:
                        net_ch = net_param.shape[1]
                        _param = torch.mean(param, dim=1, keepdim=True)
                        _param = _param.repeat(1, net_ch, 1, 1, 1)
                        pretrain_dict[k] = _param

                net_dict.update(pretrain_dict)
                self.model.load_state_dict(net_dict, strict=True)
                print('Successfully load pretrained weight')
        
        # model head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # attention
        self.q = nn.Linear(in_features, embed_size)
        self.k = nn.Linear(in_features, embed_size)
        self.v = nn.Linear(in_features, embed_size)
        
        self.attention = nn.MultiheadAttention(embed_size, num_heads=num_heads, batch_first=True) # input: (B, L, E)
        
        # classification head
        self.bowel_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.BatchNorm1d(embed_size//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(embed_size//2, out_features=1, bias=True),
        )
        
        self.ext_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.BatchNorm1d(embed_size//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(embed_size//2, out_features=1, bias=True),
        )
        
        self.kidney_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.BatchNorm1d(embed_size//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(embed_size//2, out_features=3, bias=True),
        )
        
        self.liver_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.BatchNorm1d(embed_size//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(embed_size//2, out_features=3, bias=True),
        )
        
        self.spleen_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size//2),
            nn.BatchNorm1d(embed_size//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(embed_size//2, out_features=3, bias=True),
        )
        
        # if infer, return probability
        self.infer = infer
        
    def forward(self, x, mask=None):
        # x: (b, c, d, h, w) -> (b, d, c, h, w)
        # mask (b, d): True -> ignore
        b, c, d, h, w = x.shape
        x = torch.permute(x, (0, 2, 1, 3, 4))  # (b, d, c, h, w)
        x = x.contiguous().view(b*d, c, h, w)
        feature = self.model(x)
        feature = feature.view(b, d, -1)  # (b, d, -1)
        
        # qkv
        q = self.q(feature)
        k = self.k(feature)
        v = self.v(feature)
        
        x_fea, _ = self.attention(q, k, v, key_padding_mask=mask)  # (b, d, embed_size)
        
        # classification
        out_bowel = self.bowel_fc(x_fea[:, 0, :])
        out_ext = self.ext_fc(x_fea[:, 0, :])
        out_kidney = self.kidney_fc(x_fea[:, 0, :])
        out_liver = self.liver_fc(x_fea[:, 0, :])
        out_spleen = self.spleen_fc(x_fea[:, 0, :])
        
        if self.infer:
            out_bowel = torch.sigmoid(out_bowel)
            out_ext = torch.sigmoid(out_ext)
            out_kidney = torch.softmax(out_kidney, dim=1)
            out_liver = torch.softmax(out_liver, dim=1)
            out_spleen = torch.softmax(out_spleen, dim=1)
        
        return out_bowel, out_ext, out_kidney, out_liver, out_spleen    


    
 # convert 2d model to 3d model (timm model)
def convert_2dto3d(module):
    module_output = module
    if isinstance(module, nn.Conv2d):
        w = module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0])
        module_output = nn.Conv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=False if module.bias is None else True,
            padding_mode=module.padding_mode,
        )
        module_output.weight = torch.nn.Parameter(w)
    elif isinstance(module, nn.BatchNorm2d):
        module_output = nn.BatchNorm3d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    elif isinstance(module, nn.Dropout2d):
        module_output = nn.Dropout3d(
            p=module.p,
            inplace=module.inplace,
        )
    elif isinstance(module, nn.MaxPool2d):
        module_output = nn.MaxPool3d(
            module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
            stride=module.stride if isinstance(module.stride, int) else module.stride[0],
            padding=module.padding if isinstance(module.padding, int) else module.padding[0],
            dilation=module.dilation if isinstance(module.dilation, int) else module.dilation[0],
#             return_indices=module.return_indicies,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, nn.AvgPool2d):
        module_output = nn.AvgPool3d(
            kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
            stride=module.stride if isinstance(module.stride, int) else module.stride[0],
            padding=module.padding if isinstance(module.padding, int) else module.padding[0],
            ceil_mode=module.ceil_mode,
        )
        
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_2dto3d(child)
        )
    del module
    return module_output


# competition release models
class FeatureExtractor(nn.Module):
    def __init__(self, hidden, num_channel):
        super(FeatureExtractor, self).__init__()
        import timm

        self.hidden = hidden
        self.num_channel = num_channel

        self.cnn = timm.create_model(model_name = 'regnety_002',  # 'regnety_002'
                                     pretrained = True,
                                     num_classes = 0,
                                     in_chans = num_channel)

        self.fc = nn.Linear(hidden, hidden//2)

    def forward(self, x):
        batch_size, num_frame, h, w = x.shape
        x = x.reshape(batch_size, num_frame//self.num_channel, self.num_channel, h, w)
        x = x.reshape(-1, self.num_channel, h, w)
        x = self.cnn(x)
        x = x.reshape(batch_size, num_frame//self.num_channel, self.hidden)

        x = self.fc(x)
        return x

class ContextProcessor(nn.Module):
    def __init__(self, hidden):
        super(ContextProcessor, self).__init__()
        from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel

        self.transformer = RobertaPreLayerNormModel(
            RobertaPreLayerNormConfig(
                hidden_size = hidden//2,
                num_hidden_layers = 1,
                num_attention_heads = 4,
                intermediate_size = hidden*2,
                hidden_act = 'gelu_new',
                )
            )

        del self.transformer.embeddings.word_embeddings

        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.transformer(inputs_embeds = x).last_hidden_state

        apool = torch.mean(x, dim = 1)
        mpool, _ = torch.max(x, dim = 1)
        x = torch.cat([mpool, apool], dim = -1)

        x = self.dense(x)
        x = self.activation(x)
        return x

class Custom3DCNN(nn.Module):
    def __init__(self, hidden = 368, num_channel = 2):
        super(Custom3DCNN, self).__init__()

        self.full_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)

        self.full_processor = ContextProcessor(hidden=hidden)
        self.kidney_processor = ContextProcessor(hidden=hidden)
        self.liver_processor = ContextProcessor(hidden=hidden)
        self.spleen_processor = ContextProcessor(hidden=hidden)

        self.bowel = nn.Linear(hidden, 1)
        self.extravasation = nn.Linear(hidden, 1)
        self.kidney = nn.Linear(hidden, 3)
        self.liver = nn.Linear(hidden, 3)
        self.spleen = nn.Linear(hidden, 3)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, full_input, crop_kidney, crop_liver, crop_spleen):
        full_output = self.full_extractor(full_input)
        kidney_output = self.kidney_extractor(crop_kidney)
        liver_output = self.liver_extractor(crop_liver)
        spleen_output = self.spleen_extractor(crop_spleen)

        full_output2 = self.full_processor(torch.cat([full_output, kidney_output, liver_output, spleen_output], dim = 1))
        kidney_output2 = self.kidney_processor(torch.cat([full_output, kidney_output], dim = 1))
        liver_output2 = self.liver_processor(torch.cat([full_output, liver_output], dim = 1))
        spleen_output2 = self.spleen_processor(torch.cat([full_output, spleen_output], dim = 1))

        bowel = self.bowel(full_output2)
        extravasation = self.extravasation(full_output2)
        kidney = self.kidney(kidney_output2)
        liver = self.liver(liver_output2)
        spleen = self.spleen(spleen_output2)


        # any_injury = torch.stack([
            # self.softmax(bowel)[:, 0],
            # self.softmax(extravasation)[:, 0],
            # self.softmax(kidney)[:, 0],
            # self.softmax(liver)[:, 0],
            # self.softmax(spleen)[:, 0]
        # ], dim = -1)
        # any_injury = 1 - any_injury
        # any_injury, _ = any_injury.max(1)
        return bowel, extravasation, kidney, liver, spleen #, any_injury


# one stage approach
class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        from itertools import repeat
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)



class MLPAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, attention_dim=None):
        super(MLPAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)
 
    def forward(self, x):
        """
        :param x: seq_len, batch_size, hidden_dim
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        batch_size, seq_len, _ = x.size()
        
        H = torch.tanh(self.proj_w(x)) # (batch_size, seq_len, hidden_dim)
        
        att_scores = torch.softmax(self.proj_v(H),axis=1) # (batch_size, seq_len)
        
        attn_x = (x * att_scores).sum(1) # (batch_size, hidden_dim)
        return attn_x


class RSNAClassifier(nn.Module):
    def __init__(self, model_arch, hidden_dim=128, seq_len=3, pretrained=False, infer=False):
        super().__init__()
        import timm

        self.seq_len = seq_len
        self.model_arch = model_arch
        self.model = timm.create_model(model_arch, in_chans=3, pretrained=pretrained)

        cnn_feature = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.spatialdropout = SpatialDropout(0.1)
        # self.spatialdropout = nn.Dropout2d(p=0.1, inplace=False)
        self.gru = nn.GRU(cnn_feature, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp_attention_layer = MLPAttentionNetwork(2 * hidden_dim)
        # self.logits = nn.Sequential(
            # nn.Linear(2 * hidden_dim, 13),
        # )

        # five type injury
        self.bowel_fc = nn.Linear(2*hidden_dim, out_features=1, bias=True)
        self.ext_fc = nn.Linear(2*hidden_dim, out_features=1, bias=True)
        self.kidney_fc = nn.Linear(2*hidden_dim, out_features=3, bias=True)
        self.liver_fc = nn.Linear(2*hidden_dim, out_features=3, bias=True)
        self.spleen_fc = nn.Linear(2*hidden_dim, out_features=3, bias=True)

        self.infer = infer

    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs*self.seq_len//3, 3, x.size(2), x.size(3))
        features = self.model(x)
        features = self.pooling(features).view(bs*self.seq_len//3, -1)
        features = self.spatialdropout(features) 
        # print(features.shape)
        features = features.reshape(bs, self.seq_len//3, -1) 
        features, _ = self.gru(features)            
        atten_out = self.mlp_attention_layer(features) 
        # pred = self.logits(atten_out)
        # pred = pred.view(bs, -1)
        # return pred

        # classification
        out_bowel = self.bowel_fc(atten_out)
        out_ext = self.ext_fc(atten_out)
        out_kidney = self.kidney_fc(atten_out)
        out_liver = self.liver_fc(atten_out)
        out_spleen = self.spleen_fc(atten_out)
        
        if self.infer:
            out_bowel = torch.sigmoid(out_bowel)
            out_ext = torch.sigmoid(out_ext)
            out_kidney = torch.softmax(out_kidney, dim=1)
            out_liver = torch.softmax(out_liver, dim=1)
            out_spleen = torch.softmax(out_spleen, dim=1)
        
        return out_bowel, out_ext, out_kidney, out_liver, out_spleen


# 14th place solution
class ClassificationHead(nn.Module):

    def __init__(self, input_dimensions):

        super(ClassificationHead, self).__init__()

        self.bowel_head = nn.Linear(input_dimensions, 1, bias=True)
        self.extravasation_head = nn.Linear(input_dimensions, 1, bias=True)
        self.kidney_head = nn.Linear(input_dimensions, 3, bias=True)
        self.liver_head = nn.Linear(input_dimensions, 3, bias=True)
        self.spleen_head = nn.Linear(input_dimensions, 3, bias=True)

    def forward(self, x):

        bowel_output = self.bowel_head(x)
        extravasation_output = self.extravasation_head(x)
        kidney_output = self.kidney_head(x)
        liver_output = self.liver_head(x)
        spleen_output = self.spleen_head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class Attention(nn.Module):

    def __init__(self, sequence_length, dimensions, bias=True):

        super(Attention, self).__init__()

        weight = torch.zeros(dimensions, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(sequence_length))

    def forward(self, x):

        input_batch_size, input_sequence_length, input_dimensions = x.shape

        eij = torch.mm(
            x.contiguous().view(-1, input_dimensions),
            self.weight
        ).view(-1, input_sequence_length)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        output = torch.sum(weighted_input, 1)

        return output


class MILClassificationModel(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, mil_pooling_type, feature_pooling_type, dropout_rate, freeze_parameters):

        super(MILClassificationModel, self).__init__()
        import timm

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.mil_pooling_type = mil_pooling_type
        self.feature_pooling_type = feature_pooling_type
        input_features = self.backbone.get_classifier().in_features
        self.backbone.classifier = nn.Identity()

        if self.feature_pooling_type == 'gem':
            self.pooling = GeM()
        elif self.feature_pooling_type == 'attention':
            self.pooling = nn.Sequential(
                nn.LayerNorm(normalized_shape=input_features),
                Attention(sequence_length=64, dimensions=input_features)
            )
        else:
            self.pooling = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features)

    def forward(self, x):

        input_batch_size, input_channel, input_depth, input_height, input_width = x.shape
        x = x.contiguous().view(input_batch_size * input_depth, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.mil_pooling_type == 'avg':
            x = x.contiguous().view(input_batch_size, input_depth, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.mil_pooling_type == 'max':
            x = x.contiguous().view(input_batch_size, input_depth, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)[0]
        elif self.mil_pooling_type == 'concat':
            x = x.contiguous().view(input_batch_size, input_depth * feature_channel, feature_height, feature_width)
        else:
            raise ValueError(f'Invalid MIL pooling type {self.mil_pooling_type}')

        if self.feature_pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)
        elif self.feature_pooling_type == 'gem':
            x = self.pooling(x).contiguous().view(x.size(0), -1)
        elif self.feature_pooling_type == 'attention':
            input_batch_size, feature_channel = x.shape[:2]
            x = x.contiguous().view(input_batch_size, feature_channel, -1).permute(0, 2, 1)
            x = self.pooling(x)
        else:
            raise ValueError(f'Invalid feature pooling type {self.feature_pooling_type}')

        x = self.dropout(x)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output
