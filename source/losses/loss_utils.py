import torch
from torch import nn


def focal_loss_function(hyperparameter: float):
    """
    :param hyperparameter: Scales the size of the gradient in backward
    :return: an autograd.Function ready to apply

    This autograd.Function computes the IoU in the forward pass, and has gradients with similar properties as the
    focal loss in the backward pass.
    """

    class FocalTypeLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, positives, negatives, weight):
            """
            :param logits: Any shape, dtype float
            :param positives: Same shape as logits, dtype float
            :param negatives: Same shape as logits, dtype float
            :return: scalar, an IoU type quantity

            Note that the output is only for collecting statistic. It is not differentiable, and the backward pass is
            independent of it.
            """
            predicted_positives = logits > 0
            positives_binary = positives > 0
            ctx.save_for_backward(logits, predicted_positives.float(), positives, negatives, weight)
            intersection = (predicted_positives * positives_binary).sum().float()
            union = (predicted_positives | positives_binary).sum().float().clamp(min=1)
            return intersection / union

        @staticmethod
        def backward(ctx, dummy_gradient):
            """
            :param dummy_gradient: Argument is not used
            :return: Gradient for the logits in forward.
            """
            logits, predicted_positives, positives, negatives, weight = ctx.saved_tensors

            predicted_negatives = 1 - predicted_positives
            logits_exp = logits.exp()
            target_pos_factor = 1 / (1 + logits_exp) * (-logits.clamp(min=0) ** 2 / 2).exp()
            target_neg_factor = logits_exp / (1 + logits_exp) * (-logits.clamp(max=0) ** 2 / 2).exp()

            # The four 1's here are in principle all hyperparameters, but we've found them hard to choose.
            gradient = (predicted_positives * positives * 1 * target_pos_factor
                        + predicted_negatives * positives * 1 * target_pos_factor
                        - predicted_positives * negatives * 1 * target_neg_factor
                        - predicted_negatives * negatives * 1 * target_neg_factor)
            return -gradient * hyperparameter * weight, None, None, None

    return FocalTypeLoss.apply


def weighted_kl_loss(hyperparameter: float):
    """
    :param hyperparameter: Scales the size of the gradient in backward
    :return: an autograd.Function ready to apply

    This autograd.Function computes the IoU in the forward pass, and has gradients with similar properties as the
    focal loss in the backward pass.
    """

    class WeightedKLLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, soft_mask):
            """
            :param logits: Any shape, dtype float
            :param positives: Same shape as logits, dtype float
            :param negatives: Same shape as logits, dtype float
            :return: scalar, an IoU type quantity

            Note that the output is only for collecting statistic. It is not differentiable, and the backward pass is
            independent of it.
            """
            negatives = soft_mask < 1e-6

            logits_exp = logits.exp()
            target_pos_factor = 1 / (1 + logits_exp)
            sigmoid = logits_exp * target_pos_factor

            loss = - (soft_mask * torch.log(sigmoid) + negatives * torch.log(target_pos_factor) * hyperparameter)

            ctx.save_for_backward(soft_mask, negatives, target_pos_factor, sigmoid)
            return loss.mean()

        @staticmethod
        def backward(ctx, dummy_gradient):
            """
            :param dummy_gradient: Argument is not used
            :return: Gradient for the logits in forward.
            """
            soft_mask, negatives, target_pos_factor, target_neg_factor = ctx.saved_tensors

            gradient = soft_mask * target_pos_factor - negatives * target_neg_factor * hyperparameter
            return - gradient, None, None, None

    return WeightedKLLoss.apply


def weighted_focal_loss(hyperparameter: float):
    """
    :param hyperparameter: Scales the size of the gradient in backward
    :return: an autograd.Function ready to apply

    This autograd.Function computes the IoU in the forward pass, and has gradients with similar properties as the
    focal loss in the backward pass.
    """

    class WeightedKLLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, soft_mask):
            """
            :param logits: Any shape, dtype float
            :param positives: Same shape as logits, dtype float
            :param negatives: Same shape as logits, dtype float
            :return: scalar, an IoU type quantity

            Note that the output is only for collecting statistic. It is not differentiable, and the backward pass is
            independent of it.
            """
            negatives = soft_mask < 1e-6

            logits_exp = logits.exp()
            target_pos_factor = 1 / (1 + logits_exp)
            sigmoid = logits_exp * target_pos_factor

            loss = - (soft_mask * torch.log(sigmoid) + negatives * torch.log(target_pos_factor) * hyperparameter)

            ctx.save_for_backward(soft_mask, negatives, target_pos_factor, sigmoid, logits)
            return loss.mean()

        @staticmethod
        def backward(ctx, dummy_gradient):
            """
            :param dummy_gradient: Argument is not used
            :return: Gradient for the logits in forward.
            """
            soft_mask, negatives, target_pos_factor, target_neg_factor, logits = ctx.saved_tensors

            target_pos_factor = target_pos_factor * (-logits.clamp(min=0) ** 2 / 2).exp()
            target_neg_factor = target_neg_factor * (-logits.clamp(max=0) ** 2 / 2).exp()

            gradient = soft_mask * target_pos_factor - negatives * target_neg_factor * hyperparameter
            return - gradient, None, None, None

    return WeightedKLLoss.apply


def real_focal_loss_function(hyperparameter: float):
    """
    This function is similar to focal_loss_function, but on the backward pass returns the true focal loss, with
    hyperparameter gamma=2 (default of original paper).
    It is slightly less aggressive in suppressing gradients from confident predictions than our version of the
    focal loss, and performed slightly worse in preliminary experiments.
    """

    class RealFocalLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, positives, negatives, weight):
            predicted_positives = logits > 0
            positives_binary = positives > 0
            ctx.save_for_backward(logits, predicted_positives.float(), positives, negatives, weight)
            intersection = (predicted_positives * positives_binary).sum().float()
            union = (predicted_positives | positives_binary).sum().float().clamp(min=1)
            return intersection / union

        @staticmethod
        def backward(ctx, dummy_gradient):
            logits, predicted_positives, positives, negatives, weight = ctx.saved_tensors

            predicted_negatives = 1 - predicted_positives
            logits_exp = logits.exp()
            target_pos_factor = (2 * logits_exp * (1 + 1 / logits_exp).log() + 1) / (1 + logits_exp) ** 3
            target_neg_factor = (2 / logits_exp * (1 + logits_exp).log() + 1) / (1 + 1 / logits_exp) ** 3

            gradient = (predicted_positives * positives * 1 * target_pos_factor
                        + predicted_negatives * positives * 1 * target_pos_factor
                        - predicted_positives * negatives * 1 * target_neg_factor
                        - predicted_negatives * negatives * 1 * target_neg_factor)
            return -gradient * hyperparameter * weight, None, None, None

    return RealFocalLoss.apply


def embedding_loss_function(threshold: float):
    """
    :param threshold: Distance threshold for computing masks
    :return: an autograd.Function that can be used as loss for distances

    This loss on distances pushes positives to distances closer than the threshold, and negatives to distances
    larger than the threshold, and behaves otherwise analogous to a focal loss.
    """

    class EmbeddingLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, distances, mask, weight):
            """
            :param distances: Any shape, dtype float
            :param mask: Same shape as distances, dtype bool
            :return: Scalar, and IoU

            Note that the output is only for collecting statistic. It is not differentiable, and the backward pass is
            independent of it.
            """
            predicted_mask = distances < threshold
            distances = distances / threshold
            ctx.save_for_backward(distances, mask, weight)
            intersection = (predicted_mask * mask).sum().float()
            union = (predicted_mask | mask).sum().float().clamp(min=1)
            return intersection / union

        @staticmethod
        def backward(ctx, dummy_gradient):
            """
            :param dummy_gradient: Argument is not used
            :return: Gradient for the distances in forward.
            """
            distances, mask, weight = ctx.saved_tensors
            area = mask.sum().float() / 200

            # '1.5' is chosen here because 1.5x/(1+x) ~= e^(-(x/1.5)**4) at x = 1, i.e. gradients balance at x=1.
            positive_factor = 1.5 * distances / (1 + distances)
            negative_factor = (-(distances / 1.5) ** 4).exp()
            gradient = ~mask * negative_factor - mask * positive_factor

            return -gradient / area * weight, None, None

    return EmbeddingLoss.apply


class MassLoss(torch.autograd.Function):
    """
    The backward pass of this loss function is essentially the same as FocalLoss. In the forward pass, we compute the
    covariance instead of IoU.
    """
    @staticmethod
    def forward(ctx, logits, soft_targets, weight):
        ctx.save_for_backward(logits, soft_targets, weight)
        return ((logits - logits.mean(dim=0).unsqueeze(0)) * soft_targets).mean()

    @staticmethod
    def backward(ctx, dummy_grad):
        logits, soft_targets, weight = ctx.saved_tensors
        logits_exp = logits.exp()
        target_pos_factor = 1 / (1 + logits_exp) * (-logits.clamp(min=0) ** 2 / 2).exp()
        target_neg_factor = logits_exp / (1 + logits_exp) * (-logits.clamp(max=0) ** 2 / 2).exp()
        grad = soft_targets * (target_pos_factor * (soft_targets > 0) +
                               target_neg_factor * (soft_targets < 0))

        return - grad * weight, None, None


class SmoothSquare(torch.autograd.Function):
    """
    Behaves like the square in the forward pass, but clips gradients at +-1 in the backward pass
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        x = x.clamp(min=-1, max=1)
        return grad * 2 * x


class VarianceLoss(nn.Module):
    def forward(self, scores):
        """
        :param scores:  Shape batch_size x D, dtype float
        :return: Shape BS. D times the variance of scores over dimension 1.

        Note: Gradients are clipped in backward pass.
        """
        mean = scores.mean(1).unsqueeze(1)
        return (SmoothSquare.apply(scores - mean)).sum()


class SmoothnessPenalty(nn.Module):
    """
    This loss penalizes fluctuations of a feature vector within superpixels.
    """
    def __init__(self):
        super(SmoothnessPenalty, self).__init__()
        self.loss = VarianceLoss()

    def forward(self, embedding, superpixel):
        """
        :param embedding: Shape D x H x W, dtype float
        :param superpixel: Shape H x W, dtype int
        :return: Scalar

        Each integer in the  H x W tensor superpixels denotes a region in the image found by a superpixel algorithm
        from sklearn. The loss penalizes the variance of the embedding vector over each superpixel.
        """
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(0)
        losses = []
        for i in torch.unique(superpixel):
            scores = embedding[:, superpixel == i]
            losses.append(self.loss(scores))
        return sum(losses)
