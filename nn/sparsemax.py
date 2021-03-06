import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.base_module import BaseModule

class Sparsemax(BaseModule):
    def __init__(self, num_clusters, num_neurons_per_cluster):
        super(Sparsemax, self).__init__()
        self.num_clusters = num_clusters
        self.num_neurons_per_cluster = num_neurons_per_cluster

    def forward(self, input):

        input_reshape = torch.zeros(input.size())
        input_reshape = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        #print(input_reshape)
        dim = 2

        #translate for numerical stability
        input_shift = input_reshape # - torch.max(input_reshape, dim)[0].expand_as(input_reshape)

        #sorting input in descending order
        z_sorted = torch.sort(input_shift, dim=dim, descending=True)[0]
        input_size = input_shift.size()[dim]
        range_values = Variable(torch.arange(1, input_size+1), requires_grad=False).cuda()
        range_values = range_values.expand_as(z_sorted)

        #Determine sparsity of projection
        bound = Variable(torch.zeros(z_sorted.size()),requires_grad=False).cuda()

        #z_sorted = z_sorted.type_as(bound)
        bound = 1 + torch.addcmul(bound, range_values, z_sorted)
        cumsum_zs = torch.cumsum(z_sorted, dim)
        is_gt = torch.gt(bound, cumsum_zs).type(torch.FloatTensor).cuda()
        valid = Variable(torch.zeros(range_values.size()),requires_grad=False).cuda()
        valid = torch.addcmul(valid, range_values, is_gt)
        k_max = torch.max(valid, dim)[0]
        zs_sparse = Variable(torch.zeros(z_sorted.size()),requires_grad=False).cuda()
        zs_sparse = torch.addcmul(zs_sparse, is_gt, z_sorted)
        sum_zs = (torch.sum(zs_sparse, dim) - 1)
        taus = Variable(torch.zeros(k_max.size()),requires_grad=False).cuda()
        taus = torch.addcdiv(taus, (torch.sum(zs_sparse, dim) - 1), k_max)
        taus_expanded = taus.unsqueeze(2).expand_as(input_reshape)
        output = Variable(torch.zeros(input_reshape.size())).cuda()
        output = torch.max(output, input_shift - taus_expanded)
        #self.save_for_backward(output)
        #loss = sparseMaxLoss(taus)
        return output.view(input.size())


    def backward(self, grad_output):
        #output_forward, = self.saved_tensors

        self.output = self.output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        grad_output = grad_output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        non_zeros = Variable(torch.ne(self.output, 0).type(torch.FloatTensor), requires_grad=False).cuda()
        mask_grad = Variable(torch.zeros(self.output.size()), requires_grad=False).cuda()
        mask_grad = torch.addcmul(mask_grad, non_zeros, grad_output)
        sum_mask_grad = torch.sum(mask_grad, dim)
        l1_norm_non_zeros = torch.sum(non_zeros, dim)
        sum_v = Variable(torch.zeros(sum_mask_grad.size()), requires_grad=False).cuda()
        sum_v = torch.addcdiv(sum_v, sum_mask_grad, l1_norm_non_zeros)
        self.gradInput = Variable(torch.zeros(grad_output.size()))
        self.gradInput = torch.addcmul(self.gradInput, non_zeros, grad_output - sum_v.expand_as(grad_output))
        self.gradInput = self.gradInput.view(-1, self.num_clusters*self.num_neurons_per_cluster)
        return self.gradInput
