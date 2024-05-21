import torch
import torch.nn as nn
from loss import batch_episym

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv(x) # x [32, 128, 2000, 1] -> out [32, 128, 2000, 1]) || # l1_2: x[32, 256, 2000, 1] ->[32, 128, 2000, 1]
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x) # l1_1: x [32, 128, 500, 1] - > out [32, 500, 128, 1]); # l1_2: x [32, 256, 2000, 1]
        out = out + self.conv2(out) # [32, 500, 128, 1]
        out = self.conv3(out) # [32, 128, 500, 1]
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points1),
                nn.ReLU(),
                nn.Conv2d(points1, points2, kernel_size=1),
                nn.BatchNorm2d(points2),
                nn.ReLU(),
                nn.Conv2d(points2, points1, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1 || x [32, 128, 2000, 1] -> embed [32, 500, 2000, 1]
        S = torch.softmax(embed, dim=2).squeeze(3) # S [32, 500, 2000]
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3) #out [32, 128, 500, 1]
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        #x_up: b*c*n*1 || [32, 128, 2000, 1]
        #x_down: b*c*k*1 || [32, 128, 500, 1]
        embed = self.conv(x_up)# b*k*n*1  || [32, 500, 2000, 1]
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n  ||  S [32, 500, 2000]
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)  # out [32, 128, 2000, 1]
        return out



'''
left down in UOA
'''
class left_down(nn.Module):
    def __init__(self, net_channels, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.l2 = nn.Sequential(*self.l2)

    def forward(self, x1_1):
        #x1_1: b*c*n*1
        x_down = self.down1(x1_1) # x_down [32, 128, n', 1]
        x2 = self.l2(x_down) # x2 [32, 128, n', 1] 
        return x_down, x2


'''
right up in U
'''
class right_up(nn.Module):
    def __init__(self, net_channels, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters

        self.up1 = diff_unpool(channels, l2_nums)

    def forward(self, x1_1, x2):
        #x1_1: b*c*n*1
        #x2: b*c*n'*1
        x_up = self.up1(x1_1, x2) # x_up [32, 128, n, 1]
        return x_up


class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)  

        l2_nums = 2000 # clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))
 
        self.left_down_layer1 = left_down(channels, depth, l2_nums//2)
        self.left_down_layer2 = left_down(channels, depth, l2_nums//4)
        self.left_down_layer3 = left_down(channels, depth, l2_nums//8)
        self.left_down_layer4 = left_down(channels, depth, l2_nums//16)

        self.right_up_layer1 = right_up(channels, depth, l2_nums//16)
        self.right_up_layer2 = right_up(channels, depth, l2_nums//8)
        self.right_up_layer3 = right_up(channels, depth, l2_nums//4)
        self.right_up_layer4 = right_up(channels, depth, l2_nums//2)

        self.right_cat = []
        self.right_cat.append(PointCN(2*channels, channels))
        self.right_cat = nn.Sequential(*self.right_cat)

        self.right_out = []
        for _ in range(self.layer_num//2-1):
            self.right_out.append(PointCN(channels))
        self.right_out = nn.Sequential(*self.right_out)
 

        self.l1_1 = nn.Sequential(*self.l1_1) 

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2] #32,2000
        x1_1 = self.conv1(data) #x1_1 [32, 128, 2000, 1]
        x1_1 = self.l1_1(x1_1) # [32, 128, 2000, 1] 

        # left_down & right_up in U
        x2_d1, x2_oaf1 = self.left_down_layer1(x1_1) # n -> n/2
        x2_d2, x2_oaf2 = self.left_down_layer2(x2_d1) # n/2 -> n/4
        x2_d3, x2_oaf3 = self.left_down_layer3(x2_d2) # n/4 -> n/8
        x2_d4, x2_oaf4 = self.left_down_layer4(x2_d3) # n/8 -> n/16

        x_up1 = self.right_up_layer1(x2_d3 , x2_oaf4)  # n/16 -> n/8
        x_up2 = self.right_up_layer2(x2_d2 , self.right_cat(torch.cat([x2_oaf3,x_up1], dim=1))) # n/8 -> n/4
        x_up3 = self.right_up_layer3(x2_d1 , self.right_cat(torch.cat([x2_oaf2,x_up2], dim=1))) # n/4 - > n/2
        x_up4 = self.right_up_layer4(x1_1 , self.right_cat(torch.cat([x2_oaf1,x_up3], dim=1))) # n/2 -> n
 
        x_up4s = self.right_cat(torch.cat([x1_1,x_up4], dim=1))

        out = self.right_out(x_up4s)


        logits = torch.squeeze(torch.squeeze(self.output(out),3),1) # logits [32, 2000]
        e_hat = weighted_8points(xs, logits) # [32, 9]

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]  # xs[:,0,:,:2]->x1 [32,2000,2]; xs[:,0,:,2:4]->x2  [32,2000,2]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1) #batch_episym(x1, x2, e_hat_norm)-> [32,2000]  ; residual [32, 1, 2000, 1]

        return logits, e_hat, residual


class UOANet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = OANBlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [OANBlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1  # xs [32,1,2000,4]
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2] # 32, 2000
        #data: b*1*n*c
        input = data['xs'].transpose(1,3) #input [32,4,2000,1]
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs']) #logits [32,2000], e_hat[32,9], residual [32, 1, 2000, 1]
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs']) # torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1) - > [32,6,2000,1]
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  


        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu() # [32, 9, 9]
    b, d, _ = X.size() # b=32, d=9
    bv = X.new(b,d,d) #[32, 9, 9]
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True) # e [9]  v [9, 9]
        bv[batch_idx,:,:] = v 
    bv = bv.cuda() #[32, 9, 9]
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape  # [32,1,2000,4]
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))  # [32, 2000]
    x_in = x_in.squeeze(1) # [32,2000,4]
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1) #[32, 4, 2000]

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([  # xx[:, 2]  [32, 2000]
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)  # X [32, 2000, 9]
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X  #[32, 2000, 9]
    XwX = torch.matmul(X.permute(0, 2, 1), wX)  # [32, 9, 9]
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX) #[32, 9, 9]
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9)) #v[:, :, 0] [32, 9], e_hat [32, 9]

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)  #[32, 9]
    return e_hat

