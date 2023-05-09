import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Mesh:
    x_mesh: object # (N, D), Coordinates of mesh points
    inds_me: object # (X, Ne), Indices of mesh points in each element
    x_node: object # (N, D), Coordinates of node points
    inds_ne: object # (X, Ne), Indices of node points in each element

    @property
    def n_dim(self):
        return 1 if len(self.x_node.shape) == 1 else self.x_node.shape[1]
    
    @property
    def n_elem(self):
        return self.inds_me.shape[1]


def create_segment_mesh(x_min, x_max, n_elem):
    x_mesh = torch.linspace(x_min, x_max, n_elem + 1)
    inds_me = torch.vstack([torch.arange(n_elem), torch.arange(1, n_elem + 1)])

    x_node = torch.cat([x_mesh, x_mesh[inds_me].mean(dim=0)])
    inds_ne = torch.vstack([inds_me, torch.max(inds_me) + torch.arange(inds_me.shape[1])])
    return Mesh(x_mesh, inds_me, x_node, inds_ne)


class FiniteElementNeuralNetwork(nn.Module):
    def __init__(self, mesh, basis, transform, network, boundary=None):
        super(FiniteElementNeuralNetwork, self).__init__()
        self.n_dim = mesh.n_dim
        self.inds_ne = torch.tensor(mesh.inds_ne)
        self.basis = basis
        self.transform = transform
        self.network = network
        self.boundary = boundary

    def forward(self, x_in):
        # x_in (N, B, F_in)
        # f_out (N, B, F_out)
        if self.n_dim == 1:
            x_local = self.transform.to_local_frame(x_in[..., 0])
        else:
            x_local = self.transform.to_local_frame(x_in[..., :self.n_dim])
        x_local = x_local.view(-1, self.n_dim)
        f_basis = self.basis(x_local).view(*x_in.shape[:2], -1) # (Ne, B, X)
        f_basis = torch.permute(f_basis, (2, 0, 1)).unsqueeze(-1) # (X, Ne, B)

        f_out = 0.
        for i_basis, inds in enumerate(self.inds_ne):
            f_net = self.network(x_in, inds)
            if self.boundary is not None:
                f_net = self.boundary(f_net, i_basis)
            f_out += f_net*f_basis[i_basis]
        return f_out
    

class Starter(nn.Module):
    def __init__(self, mesh, transform, batch_size, out_features):
        super(Starter, self).__init__()
        self.mesh = mesh
        self.transform = transform
        self.batch_size = batch_size
        self.out_features = out_features

    def forward(self, device):
        x_data = torch.rand(self.mesh.n_elem, self.batch_size, device=device)
        x_in = self.transform.to_global_frame(x_data).view(-1, 1)
        x_in.requires_grad = True
        x_global = x_in.view(self.mesh.n_elem, self.batch_size, 1)
        return x_in, x_global


class MultiDense(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, activation=None):
        super(MultiDense, self).__init__()
        # Initialize parameters
        params = torch.zeros([num_nodes, out_features, in_features + 1])
        nn.init.kaiming_uniform_(params[..., :in_features], math.sqrt(5.))
        self.params = nn.Parameter(params)
        # Initialize activation
        self.activation = nn.Identity() if activation is None else activation

    def forward(self, x_in, inds_ne):
        # x_in (Ne, B, F_in)
        # inds_ne (X, Ne)
        # w (Ne, F_out, F_in)
        # b (Ne, 1, F_out)
        # f_out (Ne, B, F_out)
        params = self.params[inds_ne]
        weight = params[..., :-1]
        bias = params[..., -1].unsqueeze(-2)
        return self.activation(torch.einsum('ijk,ilk->ijl', x_in, weight) + bias)
    

class MultiSequential(nn.Sequential):
    def forward(self, x_in, inds_ne):
        for module in self:
            x_in = module(x_in, inds_ne)
        return x_in
    

class FixedValueBoundary(nn.Module):
    def __init__(self, i_basis, inds_node, values):
        # inds_bd_ne (X, Ne)
        # f_bd_ne (X, Ne)
        super(FixedValueBoundary, self).__init__()
        self.i_basis = i_basis
        self.register_buffer("inds_node", inds_node)
        self.register_buffer("values", torch.atleast_2d(values))
    
    def forward(self, x_in, i_basis):
        if i_basis == self.i_basis:
            x_in[self.inds_node] = self.values
        return x_in


class LinearTransform(nn.Module):
    def __init__(self, mesh):
        super(LinearTransform, self).__init__()
        x_me = mesh.x_mesh[mesh.inds_me.T] # (Ne, E, D) or (Ne, 2) for 1D
        bias = x_me[:, None, 0] # (Ne, 1, D) or (Ne, 1) for 1D
        weight = x_me[:, 1:] - bias # (Ne, D', D), or (Ne, 1) for 1D
        self.register_buffer("bias", bias)
        self.register_buffer("weight", weight)

    def to_local_frame(self, x_in):
        # x_in (N, B)
        # weight (N, 1)
        # bias (N, 1)
        return (x_in - self.bias)/self.weight
    
    def to_global_frame(self, x_in):
        # x_in (N, B)
        # weight (N, 1)
        # bias (N, 1)
        return self.bias + self.weight*x_in
    

class SegmentBasis(nn.Module):
    def __init__(self, xl_base):
        super(SegmentBasis, self).__init__()
        self.register_buffer("coeff", compute_coeff(xl_base))

    def forward(self, x_local):
        # x_local (N, 1)
        base_vec = x_local.repeat(1, self.coeff.shape[0]) #
        base_vec[..., 0] = 1.
        base_vec = torch.cumprod(base_vec, dim=-1)
        return torch.matmul(base_vec, self.coeff)


def compute_coeff(xl_base):
    mat = base_vector(xl_base, len(xl_base) - 1).squeeze(dim=1)
    coeff = torch.linalg.inv(mat)
    return coeff


def base_vector(x_in, degree):
    # x_in (B, D)
    if len(x_in.shape) == 1:
        x_in = x_in.reshape(-1, 1)
    vec = x_in[..., None].repeat_interleave(degree + 1, dim=-1)
    vec[..., 0] = 1.
    if degree == 1:
        return vec
    return torch.cumprod(vec, dim=-1)