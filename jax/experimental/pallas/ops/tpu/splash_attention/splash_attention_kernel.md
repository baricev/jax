**[CODE_NAME]**: SegmentIds
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· q: segment identifiers for queries
· kv: segment identifiers for keys and values
**[Code Description]**: NamedTuple representing segmentation information to prevent cross-segment attention.
**[Note]**: Used to mask tokens from different segments.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: get_kernel_name
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· block_metadata: mapping with block parameters
· is_mqa: whether kernel uses MQA layout
· save_residuals: store forward residuals flag
· is_segmented: true when using segment ids
· phase: kernel phase identifier
**[Code Description]**: Builds a unique name string encoding kernel configuration for caching purposes.
**[Note]**: Phase must be 'dq', 'dkv' or 'fwd'.
**[RETURN OBJECT]**: string

**[CODE_NAME]**: _attention_reference
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: attention mask
· q: query vectors
· k: key vectors
· v: value vectors
· segment_ids: optional SegmentIds
· mask_value: value used where mask is False
· save_residuals: whether to return logsumexp
· custom_type: implementation type
· attn_logits_soft_cap: optional soft cap value
**[Code Description]**: Computes reference attention using einsum operations and optional segmentation.
**[Note]**: Returns output and residuals when save_residuals is True.
**[RETURN OBJECT]**: jax.Array or tuple with residuals

**[CODE_NAME]**: _attention_reference_default
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: boolean mask
· q: query matrix
· k: key matrix
· v: value matrix
· segment_ids: optional SegmentIds
· mask_value: fill value for masked positions
· save_residuals: flag to return logsumexp
· custom_type: unused string
· attn_logits_soft_cap: optional cap
**[Code Description]**: Implements the default attention algorithm using softmax over masked logits and computes outputs.
**[Note]**: Applies soft capping when provided.
**[RETURN OBJECT]**: attention output or tuple with residuals

**[CODE_NAME]**: attention_reference
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: attention mask
· q: query matrix
· k: key matrix
· v: value matrix
· segment_ids: optional SegmentIds
· mask_value: mask fill value
· save_residuals: return residuals flag
· custom_type: type for backward implementation
· attn_logits_soft_cap: optional cap
**[Code Description]**: Public wrapper around _attention_reference with default arguments.
**[Note]**: Jitted when called via make_attention_reference.
**[RETURN OBJECT]**: attention output or tuple with residuals

**[CODE_NAME]**: _attention_reference_custom_fwd
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: mask array
· q: query matrix
· k: key matrix
· v: value matrix
· segment_ids: optional SegmentIds
· mask_value: mask fill value
· save_residuals: boolean flag
· custom_type: forward type
· attn_logits_soft_cap: optional cap
**[Code Description]**: Forward pass for custom VJP implementation of reference attention.
**[Note]**: Disallows higher order AD when save_residuals is True.
**[RETURN OBJECT]**: output and saved intermediates

**[CODE_NAME]**: _attention_reference_custom_bwd
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask_value: mask fill value
· save_residuals: unused flag
· custom_type: backward algorithm type
· attn_logits_soft_cap: optional cap
· res: saved values from forward
· do: gradient of output
**[Code Description]**: Computes gradients with respect to q, k and v for the custom VJP path.
**[Note]**: Supports both flash and vanilla algorithms.
**[RETURN OBJECT]**: tuple matching forward arguments

**[CODE_NAME]**: attention_reference_custom
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: mask array
· q: query matrix
· k: key matrix
· v: value matrix
· segment_ids: optional SegmentIds
· mask_value: mask fill value
· save_residuals: whether to return residuals
· custom_type: select flash or vanilla backward
· attn_logits_soft_cap: optional cap
**[Code Description]**: Reference attention using custom VJP for efficient backward computation.
**[Note]**: Calls _attention_reference_custom with jax.custom_vjp.
**[RETURN OBJECT]**: attention output or tuple with residuals

**[CODE_NAME]**: make_attention_reference
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: mask definition object
· is_mqa: use multi-query attention
· backward_impl: type of backward implementation
· params: additional options
**[Code Description]**: Returns a jitted function computing reference attention with optional vmapping for MQA or grouped attention.
**[Note]**: Handles custom, custom_vanilla or vanilla backward paths.
**[RETURN OBJECT]**: callable attention function

**[CODE_NAME]**: from_head_minor
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· vals: tuple of dimensions
· layout: QKVLayout value
**[Code Description]**: Reorders dimensions depending on layout enumeration.
**[Note]**: Used to map between head-dim-minor and sequence-minor layouts.
**[RETURN OBJECT]**: tuple of reordered dimensions

**[CODE_NAME]**: _next_nonzero
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· h: head index
· i: query block index
· j: key block index
· data_next_ref: next data indices
· block_mask_ref: block mask array
· m_next_ref: next mask indices
· next_i: fetch along i dimension flag
**[Code Description]**: Determines the next nonzero block coordinates and mask control flags for sparse kernels.
**[Note]**: Supports absence of masking by returning defaults.
**[RETURN OBJECT]**: tuple(next_j, next_m, is_nonzero, should_not_mask)

**[CODE_NAME]**: _apply_mask_and_soft_cap
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· qk: raw attention logits
· mask_value: fill value for masked logits
· should_not_mask: boolean control
· mask_ref: static mask blocks
· q_sequence_ref: indices of q tokens
· q_segment_ids_ref: q segment ids
· kv_segment_ids_ref: kv segment ids
· attn_logits_soft_cap: logits cap
· k_slice: slice of keys
· k_offset: offset for key indices
· bq: block size for queries
· k_in_lanes: keys arranged in lanes flag
· mask_function: callable dynamic mask
**[Code Description]**: Applies boolean masks and optional soft capping to a block of logits, returning modified logits.
**[Note]**: Supports dynamically computed masks when mask_function is provided.
**[RETURN OBJECT]**: array or tuple with mask intermediates

**[CODE_NAME]**: flash_attention_kernel
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· data_next_ref: indices for next data blocks
· block_mask_ref: block mask flags
· mask_next_ref: indices for next mask blocks
· q_ref: query data
· k_ref: key data
· v_ref: value data
· q_segment_ids_ref: query segment ids
· kv_segment_ids_ref: key/value segment ids
· mask_ref: precomputed mask blocks
· q_sequence_ref: query sequence indices
· m_scratch_ref: scratch max buffer
· l_scratch_ref: scratch sum buffer
· o_scratch_ref: scratch output buffer
· o_ref: output buffer
· logsumexp_ref: optional logsumexp output
· mask_value: value for masked logits
· grid_width: width of kernel grid
· bq: block size for queries
· bkv: block size for keys/values
· bkv_compute: compute tile size for kv
· head_dim_v: value head dimension
· q_layout: layout enum for q
· k_layout: layout enum for k
· v_layout: layout enum for v
· attn_logits_soft_cap: optional cap
· mask_function: optional dynamic mask
**[Code Description]**: Pallas kernel performing flash attention using prefetching and masking information for sparse computation.
**[Note]**: Initializes scratch buffers and writes final output with optional logsumexp.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _splash_attention_forward
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· fwd_mask_info: MaskInfo for forward pass
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· mask_value: mask fill value
· is_mqa: multi-query attention flag
· block_sizes: BlockSizes configuration
· residual_checkpoint_name: optional checkpoint tag
· mask_function: optional dynamic mask
· save_residuals: whether to return residuals
· attn_logits_soft_cap: optional cap
· interpret: run in interpreter mode flag
**[Code Description]**: Launches flash_attention_kernel with parameters derived from BlockSizes and mask info, returning outputs and optionally residuals.
**[Note]**: Validates tensor ranks and shapes.
**[RETURN OBJECT]**: attention output or tuple with residuals

**[CODE_NAME]**: _div
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· dividend: integer numerator
· divisor: integer denominator
**[Code Description]**: Returns dividend when divisor is one, else performs integer division via lax.div.
**[Note]**: Helper for indexing calculations.
**[RETURN OBJECT]**: integer

**[CODE_NAME]**: _splash_attention_custom
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· fwd_mask_info: MaskInfo for forward pass
· dq_mask_info: MaskInfo for dq gradient
· dkv_mask_info: MaskInfo for dkv gradient
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· save_residuals: store residuals flag
· mask_value: mask fill value
· is_mqa: multi-query attention flag
· block_sizes: BlockSizes
· residual_checkpoint_name: optional checkpoint tag
· mask_function: optional mask callable
· attn_logits_soft_cap: optional cap
· interpret: interpreter mode
**[Code Description]**: Custom_vjp wrapper that forwards to _splash_attention_forward and records residuals for backward.
**[Note]**: Passes mask infos as residuals to the backward phase.
**[RETURN OBJECT]**: attention output or tuple

**[CODE_NAME]**: _splash_attention_fwd
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· fwd_mask_info: MaskInfo for forward pass
· dq_mask_info: MaskInfo for dq
· dkv_mask_info: MaskInfo for dkv
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· save_residuals: boolean flag
· mask_value: mask fill value
· is_mqa: multi-query flag
· block_sizes: BlockSizes
· residual_checkpoint_name: optional tag
· mask_function: optional mask callable
· attn_logits_soft_cap: optional cap
· interpret: interpreter mode flag
**[Code Description]**: Forward rule for custom_vjp; calls _splash_attention_forward with save_residuals=True and packages residuals for the backward.
**[Note]**: Raises if higher-order AD requested.
**[RETURN OBJECT]**: tuple(output), residuals tuple

**[CODE_NAME]**: _flash_attention_dq_kernel
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· data_next_ref: next data indices
· block_mask_ref: block mask
· mask_next_ref: next mask indices
· q_ref: query tensor
· k_ref: key tensor
· v_ref: value tensor
· q_segment_ids_ref: query segment ids
· kv_segment_ids_ref: key/value segment ids
· logsumexp_ref: logsumexp input
· do_ref: gradient of output
· di_ref: grad of intermediate m values
· mask_ref: mask blocks
· q_sequence_ref: q sequence indices
· dq_scratch_ref: scratch buffer
· dq_ref: output gradient buffer
· mask_value: fill value
· grid_width: grid width
· bq: query block size
· bkv: key/value block size
· attn_logits_soft_cap: optional cap
· q_layout: layout enum for q
· k_layout: layout enum for k
· v_layout: layout enum for v
· mask_function: optional callable
**[Code Description]**: Kernel computing gradient w.r.t. queries using prefetching and masking similar to the forward flash kernel.
**[Note]**: Uses scratch accumulation before writing final dq.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _splash_attention_bwd_dq
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· logsumexp: stored logsumexp
· do: gradient of output
· di: gradient w.r.t. intermediate
· bq: query block size
· bkv: key/value block size
· is_mqa: multi-query flag
· mask_info: MaskInfo for dq
· mask_value: mask fill value
· attn_logits_soft_cap: optional cap
· q_layout: layout enum for q
· k_layout: layout enum for k
· v_layout: layout enum for v
· mask_function: optional mask callable
· interpret: interpreter mode flag
**[Code Description]**: Launches _flash_attention_dq_kernel to compute query gradients across all heads and blocks.
**[Note]**: Validates shapes and grid sizes.
**[RETURN OBJECT]**: jax.Array with dq

**[CODE_NAME]**: _flash_attention_dkv_kernel
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· data_next_ref: next data indices
· block_mask_ref: block mask
· mask_next_ref: next mask indices
· q_ref: query tensor
· k_ref: key tensor
· v_ref: value tensor
· q_segment_ids_ref: q segment ids
· kv_segment_ids_ref: kv segment ids
· mask_ref: mask blocks
· q_sequence_ref: q sequence indices
· do_ref: gradient of output
· dq_ref: gradient w.r.t. queries
· dk_scratch_ref: scratch for dk
· dv_scratch_ref: scratch for dv
· dk_ref: output dk buffer
· dv_ref: output dv buffer
· mask_value: fill value
· grid_width: grid width
· bq: query block size
· bkv: key/value block size
· bkv_compute: compute tile size
· head_dim: key/value head dimension
· q_layout: layout enum for q
· k_layout: layout enum for k
· v_layout: layout enum for v
· attn_logits_soft_cap: optional cap
· mask_function: optional callable
**[Code Description]**: Kernel computing gradients for keys and values with masking logic similar to forward pass.
**[Note]**: Accumulates gradients in scratch buffers before final write.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _splash_attention_bwd_dkv
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· do: output gradient
· dq: gradient w.r.t queries
· bq: query block size
· bkv: key/value block size
· mask_info: MaskInfo for dkv
· mask_value: mask fill value
· attn_logits_soft_cap: optional cap
· q_layout: layout enum for q
· k_layout: layout enum for k
· v_layout: layout enum for v
· mask_function: optional mask callable
· interpret: interpreter mode flag
**[Code Description]**: Invokes _flash_attention_dkv_kernel to compute dk and dv from gradients and stored mask info.
**[Note]**: Checks tensor dimensions and grid width before execution.
**[RETURN OBJECT]**: tuple(dk, dv)

**[CODE_NAME]**: _splash_attention_bwd
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· fwd_mask_info: forward MaskInfo
· dq_mask_info: dq MaskInfo or None
· dkv_mask_info: dkv MaskInfo or None
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· save_residuals: unused flag
· mask_value: fill value
· is_mqa: multi-query flag
· block_sizes: BlockSizes
· residual_checkpoint_name: optional tag
· mask_function: optional callable
· attn_logits_soft_cap: optional cap
· interpret: interpreter mode flag
· do: gradient of output
**[Code Description]**: Full backward function computing dq, dk and dv by calling the relevant kernels depending on block sizes and mask info.
**[Note]**: Supports fused or separate backward kernels.
**[RETURN OBJECT]**: tuple(dq, dk, dv)

**[CODE_NAME]**: _splash_attention
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· fwd_mask_info: forward MaskInfo
· dq_mask_info: dq MaskInfo or None
· dkv_mask_info: dkv MaskInfo or None
· q: query tensor
· k: key tensor
· v: value tensor
· segment_ids: optional SegmentIds
· mask_value: mask fill value
· is_mqa: multi-query flag
· block_sizes: BlockSizes
· save_residuals: save residuals flag
· attn_logits_soft_cap: optional cap
· residual_checkpoint_name: optional tag
· mask_function: optional callable
· interpret: interpreter mode flag
**[Code Description]**: Dispatches to forward or backward implementations using jax.custom_vjp, managing residual checkpoints and mask info.
**[Note]**: Returns outputs directly or with residuals depending on flag.
**[RETURN OBJECT]**: attention output or tuple

**[CODE_NAME]**: _make_splash_attention
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· mask: full mask or MultiHeadMask
· block_sizes: optional BlockSizes
· is_mqa: multi-query flag
· save_residuals: save residuals flag
· mask_value: mask fill value
· attn_logits_soft_cap: optional cap
· downcast_smem_data: downcast mask info data
· head_shards: number of head shards
· q_seq_shards: number of Q shards
· residual_checkpoint_name: optional name
· interpret: interpreter mode flag
**[Code Description]**: Creates a SplashAttentionKernel instance by processing the mask into MaskInfo objects and configuring execution options.
**[Note]**: Converts NumPy masks into MultiHeadMask automatically.
**[RETURN OBJECT]**: SplashAttentionKernel instance

**[CODE_NAME]**: QKVLayout
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· HEAD_DIM_MINOR: layout enumerator
· SEQ_MINOR: layout enumerator
**[Code Description]**: Enumeration describing physical layout of Q/K/V arrays for kernel execution.
**[Note]**: JSON serializable via IntEnum.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: BlockSizes
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· block_q: query block size
· block_kv: key/value block size
· block_kv_compute: compute tile for kv
· block_q_dkv: dqv query block
· block_kv_dkv: dqv kv block
· block_kv_dkv_compute: dqv compute tile
· block_q_dq: dq query block
· block_kv_dq: dq kv block
· use_fused_bwd_kernel: use fused backward
· q_layout: layout of queries
· k_layout: layout of keys
· v_layout: layout of values
**[Code Description]**: Dataclass grouping tile sizes and layout options used by Splash attention kernels. Provides helper to check availability of backward blocks.
**[Note]**: __post_init__ fills defaults and validates settings.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: SplashAttentionKernel
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· kwargs: kernel parameters
· fwd_mask_info: forward MaskInfo
· dq_mask_info: dq MaskInfo or None
· dkv_mask_info: dkv MaskInfo or None
**[Code Description]**: Callable object wrapping compiled Splash attention kernels with pytree support and manual sharding specification.
**[Note]**: __call__ invokes _splash_attention with stored settings.
**[RETURN OBJECT]**: attention output or tuple
