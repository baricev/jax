**[CODE_NAME]**: MaskInfo
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· data_next: index of next kv block
· mask_next: index of next partial mask block
· block_mask: mask block status flags
· partial_mask_blocks: list of mixed mask blocks
· q_sequence: sequence indices for Q tokens
· is_dynamic_mask: indicates dynamic vs static mask
**[Code Description]**: Named tuple storing sparse mask data used by Splash attention kernels. Arrays reside in TPU scalar memory when provided.
**[Note]**: Shapes depend on head and sequence sharding.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _HashableNDArray
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· array: underlying NumPy array
**[Code Description]**: Wrapper allowing NumPy arrays to be used as dictionary keys by hashing their bytes.
**[Note]**: Equality uses NumPy comparison with NaN support.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _downcast_to_small_type
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· array: int32 NumPy array
**[Code Description]**: Returns the smallest integer dtype that can represent all non-negative values of the input.
**[Note]**: Validates input type and content.
**[RETURN OBJECT]**: NumPy array

**[CODE_NAME]**: _check_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: mask to validate
**[Code Description]**: Ensures each row of the mask has at least one True to avoid invalid softmax operations.
**[Note]**: Raises ValueError on failure.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _get_mask_info_for_shard
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· output_shape: shape of resulting arrays
· has_mask_next: whether to build mask_next
· mask: mask slice to process
· block_shape: pallas block shape
· coords_to_partial_mask_block_index: mapping of block coordinates
· masks_per_head_shard: masks per shard
· head_start: starting head index
· num_heads: number of heads in shard
· q_seq_start: start token index for Q slice
· q_seq_shard_size: length of Q slice
· blocked_q_seq_start: starting block index for Q slice
· is_dkv: processing dKV mask flag
**[Code Description]**: Computes data_next and mask_next arrays for a specific mask shard based on partial mask blocks.
**[Note]**: Returns zeros when the mask slice is fully empty.
**[RETURN OBJECT]**: tuple of NumPy arrays

**[CODE_NAME]**: _process_dynamic_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: dense jax.Array mask
· block_shape: pallas block shape
· is_dkv: flag for dKV processing
· downcast_smem_data: whether to shrink scalar-memory arrays
· head_shards: head shard count
· q_seq_shards: Q sequence shard count
· shrink_grid: unused optimization flag
**[Code Description]**: Converts a dynamic boolean mask into MaskInfo while optionally downcasting data for TPU memory efficiency.
**[Note]**: Validates block divisibility and mask dtype.
**[RETURN OBJECT]**: tuple of MaskInfo and None

**[CODE_NAME]**: _process_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: MultiHeadMask to process
· block_shape: pallas block size
· is_dkv: flag for dKV mask
· downcast_smem_data: downcast scalar-memory arrays
· head_shards: number of head shards
· q_seq_shards: number of Q shards
· shrink_grid: enable grid shrinking
**[Code Description]**: Converts a dense MultiHeadMask into sparse MaskInfo with caching to reduce compilation cost.
**[Note]**: Decorated with LRU cache to reuse results.
**[RETURN OBJECT]**: tuple of MaskInfo and optional mask function

**[CODE_NAME]**: _shrink_mask_info
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· block_mask: mask block array
· data_next: data_next array
· mask_next: mask_next array
· head_shards: number of shards
**[Code Description]**: Reduces MaskInfo tensors by removing fully zero columns while keeping per-shard structure.
**[Note]**: Internal helper for mask shrinking.
**[RETURN OBJECT]**: tuple of arrays

**[CODE_NAME]**: _shrink_mask_info_dkv
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· block_mask: mask block array
· data_next: data_next array
· mask_next: mask_next array
· head_shards: number of shards
**[Code Description]**: Variant of _shrink_mask_info optimized for dKV mask layout using row grouping.
**[Note]**: Operates column-wise instead of row-wise.
**[RETURN OBJECT]**: tuple of arrays

**[CODE_NAME]**: _slice_mask_info
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· block_mask: block_mask array
· data_next: data_next array
· mask_next: mask_next array
· head_shards: number of shards
· slice_function: function selecting subset
**[Code Description]**: Applies slice_function to each head shard of the provided arrays and stacks the results.
**[Note]**: Used by mask shrinking routines.
**[RETURN OBJECT]**: tuple of arrays

**[CODE_NAME]**: process_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: dense MultiHeadMask
· block_shape: pallas block size
· downcast_smem_data: whether to reduce dtype size
· head_shards: head shard count
· q_seq_shards: Q shard count
· shrink_grid: optimization flag
**[Code Description]**: Public wrapper to create MaskInfo for forward pass masks.
**[Note]**: Equivalent to _process_mask with is_dkv False.
**[RETURN OBJECT]**: tuple of MaskInfo and optional function

**[CODE_NAME]**: process_mask_dkv
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: dense MultiHeadMask
· block_shape: pallas block size
· downcast_smem_data: shrink scalar-memory arrays
· head_shards: head shard count
· q_seq_shards: Q shard count
· shrink_grid: optimization flag
**[Code Description]**: Wrapper generating MaskInfo for dKV masks.
**[Note]**: Uses is_dkv True.
**[RETURN OBJECT]**: tuple of MaskInfo and optional function

**[CODE_NAME]**: process_dynamic_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: dynamic jax.Array mask
· block_shape: pallas block size
· downcast_smem_data: shrink scalar-memory arrays
· head_shards: head shard count
· q_seq_shards: Q shard count
· shrink_grid: optimization flag
**[Code Description]**: Wrapper to process dynamic masks for the forward pass.
**[Note]**: Equivalent to _process_dynamic_mask with is_dkv False.
**[RETURN OBJECT]**: tuple of MaskInfo and None

**[CODE_NAME]**: process_dynamic_mask_dkv
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: dynamic jax.Array mask
· block_shape: pallas block size
· downcast_smem_data: shrink scalar-memory arrays
· head_shards: head shard count
· q_seq_shards: Q shard count
· shrink_grid: optimization flag
**[Code Description]**: Generates MaskInfo for dynamic dKV masks.
**[Note]**: Calls _process_dynamic_mask with is_dkv True.
**[RETURN OBJECT]**: tuple of MaskInfo and None
