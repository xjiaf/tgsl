import torch


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    """
    Apply the Gumbel-Softmax trick for a differentiable
        approximation of a discrete distribution.

    :param logits: Tensor of logits, representing raw scores before softmax.
    :param temperature: Controls the smoothness of the output distribution.
                        Lower values make the distribution more discrete.
    :param eps: A small number to prevent numerical issues.
    :return: Softmax-applied, temperature-scaled logits
        for soft selection probabilities.
    """

    # Generate uniform random numbers for Gumbel noise
    # directly in the shape of logits
    uniform_random = torch.rand(logits.shape)

    # Transform the uniform random numbers into Gumbel noise
    gumbel_noise = -torch.log(-torch.log(uniform_random + eps) + eps)

    # Combine the logits with Gumbel noise, then scale by temperature
    scaled_logits = (logits + gumbel_noise) / temperature

    # Apply softmax to convert the scaled logits into probabilities
    return torch.softmax(scaled_logits, dim=-1)
