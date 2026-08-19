"""Minimal feature window helpers from official GMFlow."""


def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        batch, height, width, channels = feature.size()
        return feature.view(
            batch, num_splits, height // num_splits,
            num_splits, width // num_splits, channels
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            batch * num_splits * num_splits,
            height // num_splits, width // num_splits, channels)
    batch, channels, height, width = feature.size()
    return feature.view(
        batch, channels, num_splits, height // num_splits,
        num_splits, width // num_splits
    ).permute(0, 2, 4, 1, 3, 5).reshape(
        batch * num_splits * num_splits, channels,
        height // num_splits, width // num_splits)


def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        batch, height, width, channels = splits.size()
        batch //= num_splits * num_splits
        return splits.view(
            batch, num_splits, num_splits, height, width, channels
        ).permute(0, 1, 3, 2, 4, 5).contiguous().view(
            batch, num_splits * height, num_splits * width, channels)
    batch, channels, height, width = splits.size()
    batch //= num_splits * num_splits
    return splits.view(
        batch, num_splits, num_splits, channels, height, width
    ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
        batch, channels, num_splits * height, num_splits * width)
