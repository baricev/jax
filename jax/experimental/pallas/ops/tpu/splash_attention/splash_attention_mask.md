**[CODE_NAME]**: Mask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· shape: mask dimensions
**[Code Description]**: Base class defining common API for splash attention masks.
**[Note]**: Subclasses implement specific mask behavior.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: make_causal_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· shape: mask shape
· offset: offset between Q and KV sequences
**[Code Description]**: Generates a boolean matrix where tokens attend only to previous or equal positions considering the offset.
**[Note]**: Negative offsets create fully masked initial rows.
**[RETURN OBJECT]**: NumPy boolean array

**[CODE_NAME]**: make_local_attention_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· shape: mask shape
· window_size: (left, right) window sizes
· offset: shift for Q indices
**[Code Description]**: Builds a mask restricting attention to a sliding window around each query token.
**[Note]**: None window limits remove corresponding constraint.
**[RETURN OBJECT]**: NumPy boolean array

**[CODE_NAME]**: make_chunk_attention_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· shape: mask shape
· chunk_size: size of each causal block
**[Code Description]**: Constructs a blockwise causal mask allowing attention within fixed-size chunks.
**[Note]**: Raises ValueError if chunk_size is non positive.
**[RETURN OBJECT]**: NumPy boolean array

**[CODE_NAME]**: make_random_mask
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· shape: mask shape
· sparsity: probability of masking
· seed: RNG seed
**[Code Description]**: Produces a random mask using a binomial distribution with given sparsity.
**[Note]**: Uses numpy's global random state.
**[RETURN OBJECT]**: NumPy boolean array

**[CODE_NAME]**: LogicalOr
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· left: first mask
· right: second mask
**[Code Description]**: Combines two masks with logical OR when indexed.
**[Note]**: Raises ValueError if shapes differ.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: LogicalAnd
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· left: first mask
· right: second mask
**[Code Description]**: Combines two masks with logical AND when indexed.
**[Note]**: Checks shape compatibility upon creation.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: MultiHeadMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· masks: sequence of per-head masks
**[Code Description]**: Provides lazy multi-head masking supporting slicing per head.
**[Note]**: Rejects empty lists and nested MultiHeadMask instances.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _ComputableMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· _shape: mask dimensions
· q_sequence: cached query indices
· mask_function: callable generating mask data
**[Code Description]**: Base class for masks computed on-the-fly by the kernel to save memory.
**[Note]**: Ensures Q sequence length divides evenly across shards.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: CausalMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· offset: Q vs KV offset
**[Code Description]**: Implements a lazy causal mask using a function comparing query and key indices.
**[Note]**: Equality and hashing consider mask shape and offset.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: ChunkedCausalMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· chunk_size: chunk length
**[Code Description]**: Generates lazy causal masks restricted within fixed-size chunks.
**[Note]**: Validates positive chunk_size.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: LocalMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· window_size: tuple defining left and right extents
· offset: Q vs KV offset
**[Code Description]**: Lazily computes a mask limiting attention to a local window around each token.
**[Note]**: Handles unlimited sides with None values.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: NumpyMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· array: dense boolean mask
**[Code Description]**: Wrapper exposing a numpy array through the mask interface.
**[Note]**: Requires a 2D boolean array.
**[RETURN OBJECT]**: None

**[CODE_NAME]**: _fill_slice
**[PARAMETERS_OR_ATTRIBUTE]**: Parameters
· inp_slice: slice object
· size: dimension length
**[Code Description]**: Normalizes slice bounds to fit within the given size.
**[Note]**: Ensures step is one or None.
**[RETURN OBJECT]**: slice object

**[CODE_NAME]**: FullMask
**[PARAMETERS_OR_ATTRIBUTE]**: Attributes
· _shape: mask dimensions
**[Code Description]**: Represents a mask where all entries are True.
**[Note]**: Returns arrays of ones for any valid slice.
**[RETURN OBJECT]**: None
