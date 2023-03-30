import torch
import torch.nn.functional as F


def _pad_mask(mask):
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask):
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def _mask_byte2long(mask):
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)


def mask_bool2long(mask):
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]


def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0


def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))

def get_mask(batch_obs):
    """
    Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
    remaining capacity. 0 = feasible, 1 = infeasible
    Forbids to visit depot twice in a row, unless all nodes have been visited
    :return:
    """

    if batch_obs['visited'].dtype == torch.uint8:
        visited_loc = batch_obs['visited'][:, :, 1:]
    else:
        visited_loc = mask_long2bool(batch_obs['visited'], n=batch_obs['demand'].size(-1))

      # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
    exceeds_cap = batch_obs['demand'][batch_obs['ids'], :] + batch_obs['used_capacity'][:, :, None] > batch_obs['capacity']
    # Nodes that cannot be visited are already visited or too much demand to be served now
    mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

    # Cannot visit the depot if just visited and still unserved nodes
    mask_depot = (batch_obs['prev_a'] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
    return torch.cat((mask_depot[:, :, None], mask_loc), -1)

